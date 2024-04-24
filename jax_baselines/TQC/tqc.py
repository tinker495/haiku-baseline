from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.losses import QuantileHuberLosses
from jax_baselines.common.utils import convert_jax, soft_update, truncated_mixture
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family


class TQC(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        n_support=25,
        delta=1.0,
        critic_num=2,
        quantile_drop=0.05,
        batch_size=32,
        policy_delay=3,
        n_step=1,
        learning_starts=1000,
        target_network_update_tau=5e-4,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        mixture_type="truncated",
        risk_avoidance=0.0,
        prioritized_replay_eps=1e-6,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        super().__init__(
            env,
            model_builder_maker,
            gamma,
            learning_rate,
            buffer_size,
            train_freq,
            gradient_steps,
            batch_size,
            n_step,
            learning_starts,
            target_network_update_tau,
            prioritized_replay,
            prioritized_replay_alpha,
            prioritized_replay_beta0,
            prioritized_replay_eps,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )

        self.name = "TQC"
        self.policy_delay = policy_delay
        self.ent_coef = ent_coef
        self.target_entropy = -2.0 * np.prod(self.action_size).astype(
            np.float32
        )  # -np.sqrt(np.prod(self.action_size).astype(np.float32))
        self.ent_coef_learning_rate = 1e-6
        self.n_support = n_support
        self.delta = delta
        self.critic_num = critic_num
        self.quantile_drop = int(max(np.round(self.critic_num * self.n_support * quantile_drop), 1))
        self.middle_support = int(np.floor(n_support / 2.0))
        self.mixture_type = mixture_type
        self.risk_avoidance = risk_avoidance

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.n_support,
            self.policy_kwargs,
        )
        self.preproc, self.actor, self.critic, self.params = model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = np.log(1e-1)
            if "_" in self.ent_coef:
                init_value = np.log(float(self.ent_coef.split("_")[1]))
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = jax.device_put(init_value)
            self.ent_coef = jnp.exp(self.log_ent_coef)
        else:
            self.ent_coef = float(self.ent_coef)

        self.opt_state = self.optimizer.init(self.params)

        self.quantile = (
            jnp.linspace(0.0, 1.0, self.n_support + 1, dtype=jnp.float32)[1:]
            + jnp.linspace(0.0, 1.0, self.n_support + 1, dtype=jnp.float32)[:-1]
        ) / 2.0  # [support]
        self.quantile = jax.device_put(jnp.expand_dims(self.quantile, axis=(0, 1))).astype(
            jnp.float32
        )  # [1 x 1 x support]

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)
        self._train_ent_coef = jax.jit(self._train_ent_coef)

    def _get_update_data(self, params, feature, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, feature)
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        log_prob = jnp.sum(
            -0.5 * (jnp.square((x_t - mu) / (std + 1e-6)) + 2 * log_std + jnp.log(2 * np.pi))
            - jnp.log(1 - jnp.square(pi) + 1e-6),
            axis=1,
            keepdims=True,
        )
        return pi, log_prob

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        std = jnp.exp(log_std)
        pi = jax.nn.tanh(mu + std * jax.random.normal(key, std.shape))
        return pi

    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
            np.mean(self.scoreque), np.mean(self.lossque)
        )

    def actions(self, obs, steps):
        if self.learning_starts < steps:
            actions = np.asarray(self._get_actions(self.params, obs, next(self.key_seq)))
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(self.worker_size, self.action_size[0]))
        return actions

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)

            (
                self.params,
                self.target_params,
                self.opt_state,
                loss,
                t_mean,
                log_prob,
                new_priorities,
            ) = self._train_step(
                self.params,
                self.target_params,
                self.opt_state,
                next(self.key_seq),
                steps,
                self.ent_coef,
                **data
            )

            if not isinstance(self.ent_coef, float):
                self.log_ent_coef, self.ent_coef = self._train_ent_coef(self.log_ent_coef, log_prob)

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
            self.summary.add_scalar("loss/ent_coef", self.ent_coef, steps)

        return loss

    def _train_step(
        self,
        params,
        target_params,
        opt_state,
        key,
        step,
        ent_coef,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        not_terminateds = 1.0 - terminateds
        key1, key2 = jax.random.split(key, 2)
        targets = self._target(params, target_params, rewards, nxtobses, not_terminateds, key1, ent_coef)
        (total_loss, (critic_loss, actor_loss, abs_error, log_prob),), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, targets, weights, key2, step, ent_coef)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = soft_update(params, target_params, self.target_network_update_tau)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        return (
            params,
            target_params,
            opt_state,
            critic_loss,
            -actor_loss,
            log_prob,
            new_priorities,
        )

    def _train_ent_coef(self, log_coef, log_prob):
        def loss(log_ent_coef, log_prob):
            return -jnp.mean(log_ent_coef * (log_prob + self.target_entropy))

        grad = jax.grad(loss)(log_coef, log_prob)
        log_coef = log_coef - self.ent_coef_learning_rate * grad
        return log_coef, jnp.exp(log_coef)

    def _loss(self, params, obses, actions, targets, weights, key, step, ent_coef):
        feature = self.preproc(params, key, obses)
        qnets = self.critic(params, key, feature, actions)
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        huber0 = QuantileHuberLosses(
            logit_valid_tile,
            jnp.expand_dims(qnets[0], axis=1),
            self.quantile,
            self.delta,
        )
        critic_loss = jnp.mean(weights * huber0)
        for q in qnets[1:]:
            critic_loss += jnp.mean(
                weights
                * QuantileHuberLosses(
                    jnp.expand_dims(q, axis=1),
                    logit_valid_tile,
                    self.quantile,
                    self.delta,
                )
            )
        policy, log_prob = self._get_update_data(params, feature, key)
        qnets_pi = self.critic(jax.lax.stop_gradient(params), key, feature, policy)
        actor_loss = jnp.mean(
            ent_coef * log_prob - jnp.mean(jnp.concatenate(qnets_pi, axis=1), axis=1)
        )
        total_loss = critic_loss + actor_loss
        return total_loss, (critic_loss, actor_loss, huber0, log_prob)

    def _target(self, params, target_params, rewards, nxtobses, not_terminateds, key, ent_coef):
        next_feature = self.preproc(target_params, key, nxtobses)
        policy, log_prob = self._get_update_data(params, self.preproc(params, key, nxtobses), key)
        qnets_pi = self.critic(target_params, key, next_feature, policy)
        if self.mixture_type == "min":
            next_q = jnp.min(jnp.stack(qnets_pi, axis=-1), axis=-1) - ent_coef * log_prob
        elif self.mixture_type == "truncated":
            next_q = truncated_mixture(qnets_pi, self.quantile_drop) - ent_coef * log_prob
        return (not_terminateds * next_q * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="TQC",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        tb_log_name = tb_log_name + "({:d})".format(self.n_support)
        if self.mixture_type == "truncated":
            tb_log_name = tb_log_name + "_truncated({:d})".format(self.quantile_drop)
        else:
            tb_log_name = tb_log_name + "_min"
        if self.risk_avoidance != 0.0:
            tb_log_name = tb_log_name + "_riskavoid{:.2f}".format(self.risk_avoidance)
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
