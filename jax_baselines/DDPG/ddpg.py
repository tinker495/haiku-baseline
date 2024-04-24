from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.schedules import LinearSchedule
from jax_baselines.common.utils import convert_jax, soft_update
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.ou_noise import OUNoise


class DDPG(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        n_step=1,
        learning_starts=1000,
        target_network_update_tau=5e-4,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
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

        self.name = "DDPG"
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction

        self.noise = OUNoise(action_size=self.action_size[0], worker_size=self.worker_size)

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        self.preproc, self.actor, self.critic, self.params = model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return self.actor(params, key, self.preproc(params, key, convert_jax(obses)))  #

    def discription(self):
        return "score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
            np.mean(self.scoreque), self.epsilon, np.mean(self.lossque)
        )

    def actions(self, obs, steps):
        if self.learning_starts < steps:
            self.epsilon = self.exploration.value(steps)
            actions = np.clip(
                np.asarray(self._get_actions(self.params, obs, None)) + self.noise() * self.epsilon,
                -1,
                1,
            )
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(self.worker_size, self.action_size[0]))
        return actions

    def test_action(self, state):
        return np.clip(
            np.asarray(self._get_actions(self.params, state, None))
            + self.noise() * self.exploration_final_eps,
            -1,
            1,
        )

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
                new_priorities,
            ) = self._train_step(self.params, self.target_params, self.opt_state, None, **data)

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)

        return loss

    def _train_step(
        self,
        params,
        target_params,
        opt_state,
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
        not_terminateds = 1.0 - terminateds
        targets = self._target(target_params, rewards, nxtobses, not_terminateds, key)
        (total_loss, (critic_loss, actor_loss, abs_error)), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, targets, weights, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = soft_update(params, target_params, self.target_network_update_tau)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        return params, target_params, opt_state, critic_loss, actor_loss, new_priorities

    def _loss(self, params, obses, actions, targets, weights, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature, actions)
        error = jnp.squeeze(vals - targets)
        critic_loss = jnp.mean(weights * jnp.square(error))
        policy = self.actor(params, key, feature)
        vals = self.critic(jax.lax.stop_gradient(params), key, feature, policy)
        actor_loss = jnp.mean(-vals)
        total_loss = critic_loss + actor_loss
        return total_loss, (critic_loss, -actor_loss, jnp.abs(error))

    def _target(self, target_params, rewards, nxtobses, not_terminateds, key):
        next_feature = self.preproc(target_params, key, nxtobses)
        next_action = self.actor(target_params, key, next_feature)
        next_q = self.critic(target_params, key, next_feature, next_action)
        return (not_terminateds * next_q * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="DDPG",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        self.exploration = LinearSchedule(
            schedule_timesteps=int(self.exploration_fraction * total_timesteps),
            initial_p=self.exploration_initial_eps,
            final_p=self.exploration_final_eps,
        )
        self.epsilon = 1.0
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
