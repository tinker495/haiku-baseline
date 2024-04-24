import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.utils import (
    convert_jax,
    get_vtrace,
    kl_divergence_continuous,
    kl_divergence_discrete,
)
from jax_baselines.IMPALA.base_class import IMPALA_Family


class IMPALA_TPPO(IMPALA_Family):
    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=0.0003,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        val_coef=0.2,
        ent_coef=0.01,
        rho_max=1.0,
        kl_range=0.05,
        kl_coef=5,
        mu_ratio=0.0,
        epoch_num=3,
        log_interval=1,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        super().__init__(
            workers,
            model_builder_maker,
            manager,
            buffer_size,
            gamma,
            lamda,
            learning_rate,
            update_freq,
            batch_size,
            sample_size,
            val_coef,
            ent_coef,
            rho_max,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )
        self.mu_ratio = mu_ratio
        self.minibatch_size = 256
        self.epoch_num = epoch_num
        self.kl_range = kl_range
        self.kl_coef = kl_coef
        self.get_memory_setup()

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space, self.action_size, self.action_type, self.policy_kwargs
        )
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.opt_state = self.optimizer.init(self.params)

        self._train_step = jax.jit(self._train_step)
        self.preprocess = jax.jit(self.preprocess)
        self._loss = (
            jax.jit(self._loss_discrete)
            if self.action_type == "discrete"
            else jax.jit(self._loss_continuous)
        )

    def train_step(self, steps):
        data = self.buffer.sample(self.sample_size)

        (
            self.params,
            self.opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            rho,
            targets,
        ) = self._train_step(
            self.params,
            self.opt_state,
            next(self.key_seq),
            self.ent_coef,
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
        )

        if steps % self.log_interval == 0:
            log_dict = {
                "loss/critic_loss": float(critic_loss),
                "loss/actor_loss": float(actor_loss),
                "loss/entropy_loss": float(entropy_loss),
                "loss/mean_rho": float(rho),
                "loss/mean_target": float(targets),
            }
            self.logger_server.log_trainer.remote(steps, log_dict)
        return critic_loss, float(rho)

    def preprocess(
        self,
        params,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncteds,
    ):
        # ((b x h x w x c), (b x n)) x worker -> (worker x b x h x w x c), (worker x b x n)
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        mu_log_prob = jnp.stack(mu_log_prob)
        rewards = jnp.stack(rewards)
        terminateds = jnp.stack(terminateds)
        truncteds = jnp.stack(truncteds)
        obses = jax.vmap(convert_jax)(obses)
        nxtobses = jax.vmap(convert_jax)(nxtobses)
        feature = jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, obses)
        value = jax.vmap(self.critic, in_axes=(None, None, 0))(params, key, feature)
        next_value = jax.vmap(self.critic, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, nxtobses),
        )
        prob, pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(params, key, feature),
            actions,
            key,
            True,
        )
        rho_raw = jnp.exp(pi_prob - mu_log_prob)
        rho = jnp.minimum(rho_raw, self.rho_max)
        c_t = self.lamda * jnp.minimum(rho, self.cut_max)
        vs = jax.vmap(get_vtrace, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
            rewards, rho, c_t, terminateds, truncteds, value, next_value, self.gamma
        )
        vs_t_plus_1 = jax.vmap(
            lambda v, nv, t: jnp.where(
                t == 1, nv, jnp.concatenate([v[1:], jnp.expand_dims(nv[-1], axis=-1)])
            ),
            in_axes=(0, 0, 0),
        )(vs, next_value, truncteds)
        adv = rewards + self.gamma * (1.0 - terminateds) * vs_t_plus_1 - value
        # adv = (adv - jnp.mean(adv,keepdims=True)) / (jnp.std(adv,keepdims=True) + 1e-6)
        adv = rho * adv
        obses = [jnp.vstack(o) for o in obses]
        actions = jnp.vstack(actions)
        vs = jnp.vstack(vs)
        prob = jnp.vstack(prob)
        pi_prob = jnp.vstack(pi_prob)
        rho = jnp.vstack(rho)
        adv = jnp.vstack(adv)
        if self.mu_ratio != 0.0:
            mu_prob = jnp.vstack(mu_log_prob)
            out_prob = jnp.log(
                self.mu_ratio * jnp.exp(mu_prob) + (1.0 - self.mu_ratio) * jnp.exp(pi_prob)
            )
            return obses, actions, vs, prob, out_prob, rho, adv
        else:
            return obses, actions, vs, prob, pi_prob, rho, adv

    def _train_step(
        self,
        params,
        opt_state,
        key,
        ent_coef,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncteds,
    ):
        obses, actions, vs, old_prob, old_act_prob, rho, adv = self.preprocess(
            params,
            key,
            obses,
            actions,
            mu_log_prob,
            rewards,
            nxtobses,
            terminateds,
            truncteds,
        )

        def i_f(idx, vals):
            params, opt_state, key, critic_loss, actor_loss, entropy_loss = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(vs.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = [o[batch_idxes] for o in obses]
            actions_batch = actions[batch_idxes]
            vs_batch = vs[batch_idxes]
            old_prob_batch = old_prob[batch_idxes]
            old_act_prob_batch = old_act_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                params, opt_state, key = updates
                obs, act, vs, old_prob, old_act_prob, adv = input
                use_key, key = jax.random.split(key)
                (total_loss, (critic_loss, actor_loss, entropy_loss),), grad = jax.value_and_grad(
                    self._loss, has_aux=True
                )(params, obs, act, vs, old_prob, old_act_prob, adv, ent_coef, use_key)
                updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state, key), (critic_loss, actor_loss, entropy_loss)

            updates, losses = jax.lax.scan(
                f,
                (params, opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    vs_batch,
                    old_prob_batch,
                    old_act_prob_batch,
                    adv_batch,
                ),
            )
            params, opt_state, key = updates
            cl, al, el = losses
            critic_loss += jnp.mean(cl)
            actor_loss += jnp.mean(al)
            entropy_loss += jnp.mean(el)
            return params, opt_state, key, critic_loss, actor_loss, entropy_loss

        val = jax.lax.fori_loop(0, self.epoch_num, i_f, (params, opt_state, key, 0.0, 0.0, 0.0))
        params, opt_state, key, critic_loss, actor_loss, entropy_loss = val
        return (
            params,
            opt_state,
            critic_loss / self.epoch_num,
            actor_loss / self.epoch_num,
            entropy_loss / self.epoch_num,
            jnp.mean(rho),
            jnp.mean(vs),
        )

    def _loss_discrete(
        self,
        params,
        obses,
        actions,
        targets,
        old_prob,
        old_act_prob,
        adv,
        ent_coef,
        key,
    ):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        ratio = jnp.exp(log_prob - old_act_prob)
        kl = jax.vmap(kl_divergence_discrete)(old_prob, prob)
        actor_loss = -jnp.mean(
            jnp.where(
                (kl >= self.kl_range) & (adv * (ratio - 1.0) > 0.0),
                adv * ratio - self.kl_coef * kl,
                adv * ratio,
            )
        )
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss + ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(
        self,
        params,
        obses,
        actions,
        targets,
        old_prob,
        old_act_prob,
        adv,
        ent_coef,
        key,
    ):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        ratio = jnp.exp(log_prob - old_act_prob)
        kl = jax.vmap(kl_divergence_continuous)(old_prob, prob)
        actor_loss = -jnp.mean(
            jnp.where(
                (kl >= self.kl_range) & (adv * (ratio - 1.0) > 0.0),
                adv * ratio - self.kl_coef * kl,
                adv * ratio,
            )
        )
        mu, log_std = prob
        entropy_loss = jnp.mean(jnp.square(mu) - log_std)
        total_loss = self.val_coef * critic_loss + actor_loss + ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=10,
        tb_log_name="IMPALA_TPPO",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        if self.mu_ratio != 0.0:
            tb_log_name += f"({self.mu_ratio:.2f})"
        super().learn(
            total_trainstep,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
