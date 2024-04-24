import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.common.utils import convert_jax, discount_with_terminated


class A2C(Actor_Critic_Policy_Gradient_Family):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=32,
        val_coef=0.2,
        ent_coef=0.5,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="rmsprop",
    ):
        super().__init__(
            env,
            model_builder_maker,
            gamma,
            learning_rate,
            batch_size,
            val_coef,
            ent_coef,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )

        self.name = "A2C"
        self.get_memory_setup()

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space, self.action_size, self.action_type, self.policy_kwargs
        )

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.opt_state = self.optimizer.init(self.params)
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
            np.mean(self.scoreque), np.mean(self.lossque)
        )

    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()

        (
            self.params,
            self.opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            targets,
        ) = self._train_step(self.params, self.opt_state, None, **data)

        if self.summary:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
            self.summary.add_scalar("loss/entropy_loss", entropy_loss, steps)
            self.summary.add_scalar("loss/mean_target", targets, steps)

        return critic_loss

    def _train_step(
        self,
        params,
        opt_state,
        key,
        obses,
        states,
        actions,
        rewards,
        ep_idx,
        terminateds,
    ):
        obses = [jnp.stack(zo) for zo in zip(*obses)]  # (worker, n + 1, *obs_shape)
        states = [jnp.stack(s) for s in zip(*states)]  # (worker, n + 1, *state_shape)
        actions = jnp.stack(actions)  # (worker, n)
        rewards = jnp.stack(rewards)  # (worker, n)
        ep_idx = jnp.stack(ep_idx)  # (worker, n+1)
        filled = jnp.not_equal(ep_idx[:, :-1], -1).astype(jnp.float32)  # (worker, n)
        dones = jnp.not_equal(ep_idx[:, 1:], ep_idx[:, :-1])  # (worker, n)
        terminateds = jnp.stack(terminateds)  # (worker, n)
        truncateds = jnp.logical_and(dones, jnp.logical_not(terminateds))  # (worker, n)
        obses = convert_jax(obses)  # (worker, n + 1, *obs_shape)
        value = jax.vmap(self.critic, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, obses),
        )
        targets = jax.vmap(discount_with_terminated, in_axes=(0, 0, 0, 0, None))(
            rewards, terminateds, truncateds, value[:, 1:], self.gamma
        )
        obses = [jnp.vstack(o[:, :-1]) for o in obses]
        actions = jnp.vstack(actions)
        value = jnp.vstack(value[:, :-1])
        targets = jnp.vstack(targets)
        filled = jnp.vstack(filled)
        adv = targets - value
        (total_loss, (critic_loss, actor_loss, entropy_loss)), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, targets, adv, filled, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss, entropy_loss, jnp.mean(targets)

    def _loss_discrete(self, params, obses, actions, targets, adv, filled, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)) * filled[:, 0])

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv) * filled)
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy * filled)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, targets, adv, filled, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)) * filled[:, 0])

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv) * filled)
        mu, log_std = prob
        entropy_loss = jnp.mean((jnp.square(mu) - log_std) * filled)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _value_loss(self, params, obses, targets, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
        return critic_loss

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="A2C",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
