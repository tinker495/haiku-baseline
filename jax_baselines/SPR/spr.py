from copy import deepcopy

import dm_pix as pix
import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.utils import convert_jax, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.SPR.efficent_buffer import (
    PrioritizedTransitionReplayBuffer,
    TransitionReplayBuffer,
)


class SPR(Q_Network_Family):
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
        double_q=False,
        dueling_model=False,
        n_step=1,
        off_policy_fix=False,
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
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):

        self.name = "SPR"
        self.shift_size = 4
        self.prediction_depth = 5
        self.off_policy_fix = off_policy_fix
        self.intensity_scale = 0.05
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

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

        self._gamma = jnp.power(self.gamma, jnp.arange(self.n_step))

        if _init_setup_model:
            self.setup_model()

    def get_memory_setup(self):
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedTransitionReplayBuffer(
                self.buffer_size,
                self.observation_space,
                1,
                prediction_depth=max(self.prediction_depth, self.n_step),
                alpha=self.prioritized_replay_alpha,
                eps=self.prioritized_replay_eps,
            )
        else:
            self.replay_buffer = TransitionReplayBuffer(
                self.buffer_size,
                self.observation_space,
                1,
                prediction_depth=max(self.prediction_depth, self.n_step),
            )

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.categorial_bar_n,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.model,
            self.transition,
            self.projection,
            self.prediction,
            self.params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.categorial_bar = jnp.expand_dims(
            jnp.linspace(self.categorial_min, self.categorial_max, self.categorial_bar_n),
            axis=0,
        )  # [1, 51]
        self._categorial_bar = jnp.expand_dims(self.categorial_bar, axis=0)  # [1, 1, 51]
        self.delta_bar = jax.device_put(
            (self.categorial_max - self.categorial_min) / (self.categorial_bar_n - 1)
        )

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.expand_dims(
            jnp.argmax(
                jnp.sum(
                    self.get_q(params, convert_jax(obses), key) * self._categorial_bar,
                    axis=2,
                ),
                axis=1,
            ),
            axis=1,
        )

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            self.train_steps_count += 1
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
                rprloss,
            ) = self._train_step(
                self.params,
                self.target_params,
                self.opt_state,
                self.train_steps_count,
                next(self.key_seq),
                **data,
            )

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/rprloss", rprloss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)

        return loss

    def _image_augmentation(self, obs, key):
        """Random augmentation for input images.

        Args:
            obses (np.ndarray): input images  B x K x H x W x C
            key (jax.random.PRNGKey): random key

        Returns:
            list(np.ndarray): augmented images
        """

        def random_shift(obs, key):  # K x H x W x C
            obs = jnp.pad(
                obs,
                ((self.shift_size, self.shift_size), (self.shift_size, self.shift_size), (0, 0)),
                mode="constant",
            )
            obs = pix.random_crop(
                key,
                obs,
                (
                    obs.shape[0] - self.shift_size * 2,
                    obs.shape[1] - self.shift_size * 2,
                    obs.shape[2],
                ),
            )
            return obs

        def Intensity(obs, key):
            noise = (
                1.0 + jnp.clip(jax.random.normal(key, (1, 1, 1)), -2.0, 2.0) * self.intensity_scale
            )
            return obs * noise

        def augment(obs, key):
            subkey1, subkey2 = jax.random.split(key)
            obs_len = obs.shape[0]
            obs = jax.vmap(random_shift)(obs, jax.random.split(subkey1, obs_len))
            obs = jax.vmap(Intensity)(obs, jax.random.split(subkey2, obs_len))
            return obs

        batch_size = obs.shape[0]
        return jax.vmap(augment)(obs, jax.random.split(key, batch_size))

    def get_last_idx(self, params, obses, actions, filled, key):
        if self.n_step == 1:
            return jnp.zeros((obses[0].shape[0], 1), jnp.int32), jnp.ones(
                (obses[0].shape[0], 1), jnp.bool_
            )
        parsed_filled = jnp.reshape(filled[:, : self.n_step], (-1, self.n_step))
        last_idxs = jnp.argmax(
            parsed_filled * jnp.arange(1, self.n_step + 1), axis=1, keepdims=True
        )
        if not self.off_policy_fix:
            return last_idxs, parsed_filled
        action_obs = [o[:, 1 : self.n_step] for o in obses]  # 1 ~ n_step
        pred_actions = jax.vmap(
            lambda o: jnp.argmax(
                jnp.sum(
                    self.get_q(params, o, key) * self._categorial_bar,
                    axis=2,
                ),
                axis=1,
            ),
            in_axes=1,
            out_axes=1,
        )(
            action_obs
        )  # B x n_step - 1
        action_equal = jnp.not_equal(
            pred_actions, jnp.squeeze(actions[:, 1 : self.n_step + 1])
        )  # B x n_step - 1
        minimum_not_equal = 1 + jnp.argmax(
            action_equal * jnp.arange(self.n_step, 1, -1), axis=1, keepdims=True
        )
        last_idxs = jnp.minimum(last_idxs, minimum_not_equal)
        # fill partial filled
        parsed_filled = jnp.less_equal(jnp.arange(0, self.n_step), last_idxs)
        return last_idxs, parsed_filled

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
        terminateds,
        filled,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        actions = actions.astype(jnp.int32)
        not_terminateds = 1.0 - terminateds
        obses = [
            jax.lax.cond(
                len(o.shape) >= 5,
                lambda: self._image_augmentation(o, key),
                lambda: o,
            )
            for o in obses
        ]
        parsed_obses = [jnp.reshape(o[:, 0], (-1, *o.shape[2:])) for o in obses]
        last_idxs, parsed_filled = self.get_last_idx(params, obses, actions, filled, key)
        parsed_nxtobses = [
            jnp.reshape(
                jnp.take_along_axis(o, jnp.reshape(last_idxs + 1, (-1, 1, 1, 1, 1)), axis=1),
                (-1, *o.shape[2:]),
            )
            for o in obses
        ]
        parsed_actions = jnp.reshape(actions[:, 0], (-1, 1, 1))
        rewards = jnp.reshape(rewards[:, : self.n_step], (-1, self.n_step))
        parsed_rewards = jnp.sum(rewards * self._gamma * parsed_filled, axis=1, keepdims=True)
        parsed_not_terminateds = jnp.take_along_axis(not_terminateds, last_idxs, axis=1)
        parsed_gamma = (
            jnp.take_along_axis(jnp.expand_dims(self._gamma, 0), last_idxs, axis=1) * self.gamma
        )
        target_distribution = self._target(
            params,
            target_params,
            parsed_obses,
            parsed_actions,
            parsed_rewards,
            parsed_nxtobses,
            parsed_not_terminateds,
            parsed_gamma,
            key,
        )
        transition_obses = [o[:, : (self.prediction_depth + 1)] for o in obses]
        transition_actions = actions[:, : self.prediction_depth]
        transition_filled = filled[:, : self.prediction_depth]
        (loss, (centropy, qloss, rprloss)), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params,
            target_params,
            transition_obses,
            transition_actions,
            transition_filled,
            parsed_obses,
            parsed_actions,
            target_distribution,
            weights,
            key,
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        target_params = params
        params = optax.apply_updates(params, updates)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = centropy
        return (
            params,
            target_params,
            opt_state,
            qloss,
            jnp.mean(
                jnp.sum(
                    target_distribution * self.categorial_bar,
                    axis=1,
                )
            ),
            new_priorities,
            rprloss,
        )

    def _loss(
        self,
        params,
        target_params,
        obses,
        actions,
        filled,
        parsed_obses,
        parsed_actions,
        target_distribution,
        weights,
        key,
    ):
        rprloss = self._represetation_loss(params, target_params, obses, actions, filled, key)
        distribution = jnp.squeeze(
            jnp.take_along_axis(self.get_q(params, parsed_obses, key), parsed_actions, axis=1)
        )
        centropy = jnp.sum(
            target_distribution * (-jnp.log(distribution + 1e-8)), axis=1
        )  # jnp.mean(jnp.sum(jnp.square(error) * filled, axis=-1) / jnp.sum(filled, axis=-1) * weights)
        mean_KLdiv = jnp.mean(centropy * weights)
        total_loss = mean_KLdiv + rprloss
        return total_loss, (
            centropy,
            mean_KLdiv,
            rprloss,
        )  # jnp.sum(jnp.abs(error) * filled, axis=-1) / jnp.sum(filled, axis=-1)

    def _represetation_loss(self, params, target_params, obses, actions, filled, key):
        initial_obs = [o[:, 0] for o in obses]  # B x H x W x C
        target_obs = [o[:, 1:] for o in obses]  # B x K x H x W x C
        initial_features = self.preproc(params, key, initial_obs)  # B x D
        target_features = jax.vmap(self.preproc, in_axes=(None, None, 1), out_axes=1)(
            target_params, key, target_obs
        )  # B x K x D
        traget_projection = jax.vmap(self.projection, in_axes=(None, None, 1), out_axes=1)(
            target_params, key, target_features
        )  # B x K x D
        traget_projection = traget_projection / jnp.linalg.norm(
            traget_projection, axis=-1, keepdims=True
        )  # normalize

        def body(carry, x):
            loss, current_features = carry
            action, filled, target_projection = x
            action = jax.nn.one_hot(jnp.squeeze(action), self.action_size[0])
            current_features = self.transition(params, key, current_features, action)
            current_projection = self.projection(params, key, current_features)
            current_projection = self.prediction(params, key, current_projection)
            current_projection = current_projection / jnp.linalg.norm(
                current_projection, axis=-1, keepdims=True
            )  # normalize
            cosine_similarity = jnp.sum(current_projection * target_projection, axis=-1) - 1.0
            loss = loss - cosine_similarity * filled
            return (loss, current_features), None

        actions = jnp.swapaxes(actions, 0, 1)
        filled = jnp.swapaxes(filled, 0, 1)
        traget_projection = jnp.swapaxes(traget_projection, 0, 1)
        (loss, _), _ = jax.lax.scan(
            body,
            (jnp.zeros((initial_features.shape[0],)), initial_features),
            (actions, filled, traget_projection),
        )
        return jnp.mean(loss)

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, gammas, key
    ):
        next_q = self.get_q(target_params, nxtobses, key)
        if self.double_q:
            next_action_q = jnp.sum(
                self.get_q(params, nxtobses, key) * self._categorial_bar, axis=2
            )
        else:
            next_action_q = jnp.sum(next_q * self._categorial_bar, axis=2)
        next_actions = jnp.expand_dims(jnp.argmax(next_action_q, axis=1), axis=(1, 2))
        next_distribution = jnp.squeeze(jnp.take_along_axis(next_q, next_actions, axis=1))

        if self.munchausen:
            next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)
            next_categorial = self.categorial_bar - jnp.sum(
                pi_next * tau_log_pi_next, axis=1, keepdims=True
            )

            q_k_targets = jnp.sum(
                self.get_q(target_params, obses, key) * self.categorial_bar, axis=2
            )
            q_sub_targets, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            log_pi = q_sub_targets - self.munchausen_entropy_tau * tau_log_pi
            munchausen_addon = jnp.take_along_axis(log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            next_categorial = self.categorial_bar
        target_categorial = gammas * not_terminateds * next_categorial + rewards  # [32, 51]
        Tz = jnp.clip(
            target_categorial, self.categorial_min, self.categorial_max
        )  # clip to range of bar
        C51_B = ((Tz - self.categorial_min) / self.delta_bar).astype(
            jnp.float32
        )  # bar index as float
        C51_L = jnp.floor(C51_B).astype(jnp.int32)  # bar lower index as int
        C51_H = jnp.ceil(C51_B).astype(jnp.int32)  # bar higher index as int
        C51_L = jnp.where((C51_H > 0) * (C51_L == C51_H), C51_L - 1, C51_L)  # C51_L.at[].add(-1)
        C51_H = jnp.where(
            (C51_L < (self.categorial_bar_n - 1)) * (C51_L == C51_H), C51_H + 1, C51_H
        )  # C51_H.at[].add(1)

        def tdist(next_distribution, C51_L, C51_H, C51_b):
            target_distribution = jnp.zeros((self.categorial_bar_n))
            target_distribution = target_distribution.at[C51_L].add(
                next_distribution * (C51_H.astype(jnp.float32) - C51_b)
            )
            target_distribution = target_distribution.at[C51_H].add(
                next_distribution * (C51_b - C51_L.astype(jnp.float32))
            )
            return target_distribution

        target_distribution = jax.vmap(tdist, in_axes=(0, 0, 0, 0))(
            next_distribution, C51_L, C51_H, C51_B
        )
        return target_distribution

    def tb_log_name_update(self, tb_log_name):
        if self.munchausen:
            tb_log_name = "M-" + tb_log_name
        if self.param_noise:
            tb_log_name = "Noisy_" + tb_log_name
        if self.dueling_model:
            tb_log_name = "Dueling_" + tb_log_name
        if self.double_q:
            tb_log_name = "Double_" + tb_log_name
        if self.n_step_method:
            if self.off_policy_fix:
                n_step_str = f"{self.n_step}~1Step_"
            else:
                n_step_str = f"{self.n_step}Step_"
            tb_log_name = n_step_str + tb_log_name
        if self.prioritized_replay:
            tb_log_name = tb_log_name + "+PER"
        return tb_log_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="SPR",
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
