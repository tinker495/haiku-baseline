from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.base_classes import select_optimizer
from jax_baselines.common.losses import FQFQuantileLosses, QuantileHuberLosses
from jax_baselines.common.utils import convert_jax, hard_update
from jax_baselines.DQN.base_class import Q_Network_Family


class FQF(Q_Network_Family):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        exploration_fraction=0.3,
        n_support=32,
        delta=1.0,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        double_q=True,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):
        super().__init__(
            env,
            model_builder_maker,
            gamma,
            learning_rate,
            buffer_size,
            exploration_fraction,
            exploration_final_eps,
            exploration_initial_eps,
            train_freq,
            gradient_steps,
            batch_size,
            double_q,
            dueling_model,
            n_step,
            learning_starts,
            target_network_update_freq,
            prioritized_replay,
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

        self.name = "FQF"
        self.n_support = n_support
        self.delta = delta
        self.fqf_factor = 1e-4
        self.ent_coef = 0.01

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.model_bulder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.n_support,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.fpf, self.params, self.fqf_params = self.model_bulder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.fqf_optimizer = select_optimizer(
            "rmsprop", self.learning_rate * self.fqf_factor, grad_max=5.0
        )
        self.fqf_opt_state = self.fqf_optimizer.init(self.fqf_params)

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def actions(self, obs, epsilon):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    self.params,
                    self.fqf_params,
                    obs,
                    next(self.key_seq) if self.param_noise else None,
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def _get_actions(self, params, fqf_params, obses, key=None) -> jnp.ndarray:
        feature = self.preproc(params, key, convert_jax(obses))
        tau, tau_hat, _ = self.fpf(fqf_params, key, feature)
        return jnp.argmax(self.get_q(params, feature, tau, tau_hat, key), axis=1, keepdims=True)

    def get_quantile(self, params, feature, tau_hat, key=None) -> jnp.ndarray:
        return self.model(params, key, feature, tau_hat)

    def quantiles_to_q(self, quantiles, tau):
        tau = jnp.expand_dims(tau, axis=1)
        q = (tau[:, :, 1:] - tau[:, :, :-1]) * quantiles
        return jnp.sum(q, axis=2)

    def get_q(self, params, feature, tau, tau_hat, key=None) -> jnp.ndarray:
        quanile_hat = self.model(params, key, feature, tau_hat)
        tau = jnp.expand_dims(tau, axis=1)
        q = (tau[:, :, 1:] - tau[:, :, :-1]) * quanile_hat
        return jnp.sum(q, axis=2)

    def train_step(self, steps, gradient_steps):
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)

            (
                self.params,
                self.fqf_params,
                self.target_params,
                self.opt_state,
                self.fqf_opt_state,
                loss,
                fqf_loss,
                t_mean,
                t_std,
                tau,
                new_priorities,
            ) = self._train_step(
                self.params,
                self.fqf_params,
                self.target_params,
                self.opt_state,
                self.fqf_opt_state,
                self.train_steps_count,
                next(self.key_seq),
                **data
            )

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/fqf_loss", fqf_loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
            self.summary.add_scalar("loss/target_stds", t_std, steps)
            self.summary.add_histogram("loss/tau", tau, steps)

        return loss

    def _train_step(
        self,
        params,
        fqf_params,
        target_params,
        opt_state,
        fqf_opt_state,
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
        (
            loss,
            (abs_error, feature, taus, tau_hats, theta_loss_tile, targets),
        ), grad = jax.value_and_grad(self._loss, has_aux=True,)(
            params,
            fqf_params,
            target_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            weights,
            key,
        )
        fqf_loss, grad_fqf = jax.value_and_grad(self._fqf_loss)(
            fqf_params, params, feature, actions, theta_loss_tile, key
        )
        fqf_update, fqf_opt_state = self.fqf_optimizer.update(
            grad_fqf, fqf_opt_state, params=fqf_params
        )
        fqf_params = optax.apply_updates(fqf_params, fqf_update)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        return (
            params,
            fqf_params,
            target_params,
            opt_state,
            fqf_opt_state,
            loss,
            fqf_loss,
            jnp.mean(jnp.sum((taus[:, 1:] - taus[:, :-1]) * targets, axis=1)),
            jnp.mean(jnp.std(targets, axis=1)),
            tau_hats,
            new_priorities,
        )

    def _loss(
        self,
        params,
        fqf_params,
        target_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        weights,
        key,
    ):
        feature = self.preproc(params, key, obses)
        taus, tau_hats, _ = self.fpf(fqf_params, key, jax.lax.stop_gradient(feature))
        targets = jax.lax.stop_gradient(
            self._target(
                params,
                fqf_params,
                target_params,
                tau_hats,
                obses,
                actions,
                rewards,
                nxtobses,
                not_terminateds,
                key,
            )
        )
        tau_hats = jax.lax.stop_gradient(tau_hats)
        theta_loss_tile = jnp.take_along_axis(
            self.get_quantile(params, feature, tau_hats, key),
            actions,
            axis=1,
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        hubber = QuantileHuberLosses(
            logit_valid_tile,
            theta_loss_tile,
            jnp.expand_dims(tau_hats, axis=1),
            self.delta,
        )
        return jnp.mean(hubber * weights), (
            hubber,
            feature,
            taus,
            tau_hats,
            theta_loss_tile,
            targets,
        )

    def _fqf_loss(self, fqf_params, params, feature, actions, tau_hat_vals, key):
        feature = jax.lax.stop_gradient(feature)
        tau, _, entropy = self.fpf(fqf_params, key, feature)
        tau_vals = jnp.take_along_axis(
            self.get_quantile(params, feature, tau[:, 1:-1], key), actions, axis=1
        )  # batch x 1 x support
        tau_vals = jax.lax.stop_gradient(jnp.squeeze(tau_vals))
        tau_hat_vals = jax.lax.stop_gradient(jnp.squeeze(tau_hat_vals))
        quantile_loss = jnp.mean(
            FQFQuantileLosses(
                tau_vals,
                tau_hat_vals,
                tau,
            )
        )
        entropy_loss = -self.ent_coef * jnp.mean(entropy)
        loss = quantile_loss + entropy_loss
        return loss

    def _target(
        self,
        params,
        fqf_params,
        target_params,
        tau_hats,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        if self.double_q:
            feature = self.preproc(params, key, nxtobses)
            _tau, _tau_hats, _ = self.fpf(fqf_params, key, feature)
            next_q = self.get_q(
                params,
                feature,
                _tau,
                _tau_hats,
                key,
            )
            feature = self.preproc(target_params, key, nxtobses)
        else:
            feature = self.preproc(target_params, key, nxtobses)
            _tau, _tau_hats, _ = self.fpf(fqf_params, key, feature)
            next_q = self.get_q(
                target_params,
                feature,
                _tau,
                _tau_hats,
                key,
            )
        tau_hats = jax.random.uniform(key, (self.batch_size, self.n_support))
        next_quantiles = self.get_quantile(target_params, feature, tau_hats, key)

        if self.munchausen:
            logsum = jax.nn.logsumexp(
                (next_q - jnp.max(next_q, axis=1, keepdims=True)) / self.munchausen_entropy_tau,
                axis=1,
                keepdims=True,
            )  # []
            tau_log_pi_next = jnp.expand_dims(
                next_q
                - jnp.max(next_q, axis=1, keepdims=True)
                - self.munchausen_entropy_tau * logsum,
                axis=2,
            )
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_q / self.munchausen_entropy_tau, axis=1),
                axis=2,
            )
            next_vals = (
                jnp.sum(
                    pi_next * (next_quantiles - jnp.expand_dims(tau_log_pi_next, axis=2)),
                    axis=1,
                )
                * not_terminateds
            )

            q_k_targets = self.get_q(
                target_params,
                self.preproc(target_params, key, obses),
                _tau,
                tau_hats,
                key,
            )
            v_k_target = jnp.max(q_k_targets, axis=1, keepdims=True)
            logsum = jax.nn.logsumexp(
                (q_k_targets - v_k_target) / self.munchausen_entropy_tau,
                axis=1,
                keepdims=True,
            )
            log_pi = jnp.expand_dims(
                q_k_targets - v_k_target - self.munchausen_entropy_tau * logsum, axis=2
            )
            munchausen_addon = jnp.take_along_axis(log_pi, actions, axis=1)

            rewards += self.munchausen_alpha * jnp.clip(munchausen_addon, a_min=-1, a_max=0)
        else:
            next_actions = jnp.expand_dims(jnp.argmax(next_q, axis=1), axis=(1, 2))
            next_vals = not_terminateds * jnp.squeeze(
                jnp.take_along_axis(next_quantiles, next_actions, axis=1)
            )  # batch x support
        return (next_vals * self._gamma) + rewards  # batch x support

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="FQF",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        tb_log_name = tb_log_name + "({:d})".format(self.n_support)
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
