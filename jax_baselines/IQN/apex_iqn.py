from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.common.losses import QuantileHuberLosses
from jax_baselines.common.utils import convert_jax, hard_update, key_gen, q_log_pi


class APE_X_IQN(Ape_X_Family):
    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_initial_eps=0.8,
        exploration_decay=0.7,
        batch_size=32,
        double_q=False,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        gradient_steps=1,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        CVaR=1.0,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        n_support=200,
        delta=0.1,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):
        super().__init__(
            workers,
            model_builder_maker,
            manager,
            gamma,
            learning_rate,
            buffer_size,
            exploration_initial_eps,
            exploration_decay,
            batch_size,
            double_q,
            dueling_model,
            n_step,
            learning_starts,
            target_network_update_freq,
            gradient_steps,
            prioritized_replay_alpha,
            prioritized_replay_beta0,
            prioritized_replay_eps,
            param_noise,
            munchausen,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
            compress_memory,
        )

        self.n_support = n_support
        self.delta = delta
        self.CVaR = CVaR
        self.risk_avoid = CVaR != 1.0

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)
        self.actor_builder = self.get_actor_builder()

        self.get_q = jax.jit(self.get_q)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def get_actor_builder(self):
        gamma = self._gamma
        action_size = self.action_size[0]
        param_noise = self.param_noise
        delta = self.delta
        n_support = self.n_support

        def builder():
            import random

            key_seq = key_gen(random.randint(0, 1000000))

            def get_abs_td_error(
                model, preproc, params, obses, actions, rewards, nxtobses, terminateds, key
            ):
                key1, key2, key3 = jax.random.split(key, 3)
                conv_obses = convert_jax(obses)
                batch_size = conv_obses[0].shape[0]
                tau = jax.random.uniform(key1, (batch_size, n_support))
                next_tau = jax.random.uniform(key2, (batch_size, n_support))
                q_values = jnp.take_along_axis(
                    model(params, key3, preproc(params, key3, conv_obses), tau),
                    jnp.expand_dims(actions.astype(jnp.int32), axis=2),
                    axis=1,
                )
                next_q = model(params, key3, preproc(params, key3, convert_jax(nxtobses)), next_tau)
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.mean(next_q, axis=2), axis=1), axis=(1, 2)
                )
                next_vals = jnp.squeeze(
                    jnp.take_along_axis(next_q, next_actions, axis=1)
                )  # batch x support
                target = rewards + gamma * (1.0 - terminateds) * next_vals
                loss = QuantileHuberLosses(q_values, jnp.expand_dims(target, axis=2), tau, delta)
                return jnp.squeeze(loss)

            def actor(model, preproc, params, obses, key):
                tau = jax.random.uniform(key, (1, n_support))
                q_values = model(params, key, preproc(params, key, convert_jax(obses)), tau)
                return jnp.expand_dims(jnp.argmax(jnp.mean(q_values, axis=2), axis=1), axis=1)

            if param_noise:

                def get_action(actor, params, obs, epsilon, key):
                    return int(np.asarray(actor(params, obs, key))[0])

            else:

                def get_action(actor, params, obs, epsilon, key):
                    if epsilon <= np.random.uniform(0, 1):
                        actions = int(np.asarray(actor(params, obs, key))[0])
                    else:
                        actions = np.random.choice(action_size)
                    return actions

            def random_action(params, obs, epsilon, key):
                return np.random.choice(action_size)

            return get_abs_td_error, actor, get_action, random_action, key_seq

        return builder

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)

            (
                self.params,
                self.target_params,
                self.opt_state,
                loss,
                t_mean,
                new_priorities,
            ) = self._train_step(
                self.params,
                self.target_params,
                self.opt_state,
                steps,
                next(self.key_seq) if self.param_noise else None,
                **data
            )

            self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if steps % self.log_interval == 0:
            log_dict = {"loss/qloss": float(loss), "loss/targets": float(t_mean)}
            self.logger_server.log_trainer.remote(steps, log_dict)

        return loss

    def _train_step(
        self,
        params,
        target_params,
        opt_state,
        steps,
        key,
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
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        key1, key2 = jax.random.split(key, 2)
        targets = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key1
        )
        (loss, abs_error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, targets, weights, key2
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = abs_error
        return params, target_params, opt_state, loss, jnp.mean(targets), new_priorities

    def _loss(self, params, obses, actions, targets, weights, key):
        tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, obses, tau, key), actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        loss = QuantileHuberLosses(
            theta_loss_tile, logit_valid_tile, jnp.expand_dims(tau, axis=1), self.delta
        )
        return jnp.mean(loss * weights), loss

    def _target(self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key):
        target_tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        next_q = self.get_q(target_params, nxtobses, target_tau, key)

        if self.munchausen:
            if self.double_q:
                next_q_mean = jnp.mean(self.get_q(params, nxtobses, target_tau, key), axis=2)
            else:
                next_q_mean = jnp.mean(next_q, axis=2)
            next_sub_q, tau_log_pi_next = q_log_pi(next_q_mean, self.munchausen_entropy_tau)
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau), axis=2
            )
            next_vals = (
                jnp.sum(
                    pi_next * (next_q - jnp.expand_dims(tau_log_pi_next, axis=2)),
                    axis=1,
                )
                * not_terminateds
            )

            q_k_targets = jnp.mean(self.get_q(target_params, obses, target_tau, key), axis=2)
            q_sub_targets, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            log_pi = q_sub_targets - self.munchausen_entropy_tau * tau_log_pi
            munchausen_addon = jnp.take_along_axis(log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            if self.double_q:
                next_actions = jnp.argmax(
                    jnp.mean(self.get_q(params, nxtobses, target_tau, key), axis=2, keepdims=True),
                    axis=1,
                    keepdims=True,
                )
            else:
                next_actions = jnp.argmax(
                    jnp.mean(next_q, axis=2, keepdims=True), axis=1, keepdims=True
                )
            next_vals = not_terminateds * jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (next_vals * self._gamma) + rewards  # batch x support

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="Ape_X_QRDQN",
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
