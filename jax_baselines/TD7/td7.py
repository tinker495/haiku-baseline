from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import trange

from jax_baselines.common.base_classes import TensorboardWriter
from jax_baselines.common.losses import hubberloss
from jax_baselines.common.utils import add_hparams, convert_jax, hard_update
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family


class TD7(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env,
        eval_env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        target_action_noise_mul=2.0,
        action_noise=0.1,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        policy_delay=2,
        learning_starts=1000,
        target_network_update_freq=250,
        prioritized_replay_alpha=0.4,
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
            1,
            learning_starts,
            0,
            True,
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

        self.name = "TD7"
        self.action_noise = action_noise
        self.target_action_noise = action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5  # self.target_action_noise*1.5
        self.target_network_update_freq = target_network_update_freq
        self.policy_delay = policy_delay

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        self.steps_before_checkpointing = int(75e4)
        self.max_eps_before_checkpointing = 25

        self.eval_env = eval_env
        self.eval_freq = self.steps_before_checkpointing // 5
        self.eval_eps = 10

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.encoder,
            self.action_encoder,
            self.actor,
            self.critic,
            self.encoder_params,
            self.params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.fixed_encoder_params = deepcopy(self.encoder_params)
        self.fixed_encoder_target_params = deepcopy(self.encoder_params)
        self.target_params = deepcopy(self.params)
        self.checkpoint_encoder_params = deepcopy(self.encoder_params)
        self.checkpoint_params = deepcopy(self.params)

        self.params["values"] = {
            "min_value": jnp.array([np.inf], dtype=jnp.float32),
            "max_value": jnp.array([-np.inf], dtype=jnp.float32),
        }
        self.target_params["values"] = {
            "min_value": jnp.array([0], dtype=jnp.float32),
            "max_value": jnp.array([0], dtype=jnp.float32),
        }

        self.encoder_opt_state = self.optimizer.init(self.encoder_params)
        self.opt_state = self.optimizer.init(self.params)
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def _get_actions(self, encoder_params, params, obses, key=None) -> jnp.ndarray:
        feature = self.preproc(encoder_params, key, convert_jax(obses))
        zs = self.encoder(encoder_params, key, feature)
        return self.actor(params, key, feature, zs)

    def actions(self, obs, steps, use_checkpoint=False, exploration=True):
        if self.learning_starts < steps:
            if use_checkpoint:
                actions = np.asarray(
                    self._get_actions(
                        self.checkpoint_encoder_params, self.checkpoint_params, obs, None
                    )
                )
            else:
                actions = np.asarray(
                    self._get_actions(self.fixed_encoder_params, self.params, obs, None)
                )
            if exploration:
                actions = np.clip(
                    actions
                    + self.action_noise * np.random.normal(
                        0, 1, size=(self.worker_size, self.action_size[0])
                    ),
                    -1,
                    1,
                )
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(self.worker_size, self.action_size[0]))
        return actions

    def end_episode(self, steps, score, eplen):
        if self.learning_starts > steps:
            return
        self.eps_since_update += 1
        self.timesteps_since_update += eplen

        self.min_return = min(self.min_return, score)
        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset(steps)

        # Update checkpoint
        elif self.eps_since_update >= self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_params = self.params
            self.checkpoint_encoder_params = self.fixed_encoder_params
            self.train_and_reset(steps)

    def train_and_reset(self, steps):
        if steps > self.steps_before_checkpointing:
            self.max_eps_before_update = self.max_eps_before_checkpointing
        self.loss_mean = self.train_step(steps, self.timesteps_since_update * self.gradient_steps)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        repr_losses = []
        losses = []
        targets = []
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)

            (
                self.encoder_params,
                self.params,
                self.fixed_encoder_params,
                self.fixed_encoder_target_params,
                self.target_params,
                self.encoder_opt_state,
                self.opt_state,
                repr_loss,
                loss,
                t_mean,
                new_priorities,
            ) = self._train_step(
                self.encoder_params,
                self.params,
                self.fixed_encoder_params,
                self.fixed_encoder_target_params,
                self.target_params,
                self.encoder_opt_state,
                self.opt_state,
                next(self.key_seq),
                self.train_steps_count,
                **data,
            )
            repr_losses.append(repr_loss)
            losses.append(loss)
            targets.append(t_mean)

            self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        mean_repr_loss = jnp.mean(jnp.array(repr_losses))
        mean_loss = jnp.mean(jnp.array(losses))
        mean_target = jnp.mean(jnp.array(targets))

        if self.summary:
            self.summary.add_scalar("loss/encoder_loss", mean_repr_loss, steps)
            self.summary.add_scalar("loss/qloss", mean_loss, steps)
            self.summary.add_scalar("loss/targets", mean_target, steps)
            self.summary.add_scalar("loss/min_value", self.params["values"]["min_value"], steps)
            self.summary.add_scalar("loss/max_value", self.params["values"]["max_value"], steps)

        return mean_loss

    def _train_step(
        self,
        encoder_params,
        params,
        fixed_encoder_params,
        fixed_encoder_target_params,
        target_params,
        encoder_opt_state,
        opt_state,
        key,
        step,
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

        repr_loss, grad = jax.value_and_grad(self._encoder_loss)(
            encoder_params, obses, nxtobses, actions, key
        )
        updates, encoder_opt_state = self.optimizer.update(
            grad, encoder_opt_state, params=encoder_params
        )
        encoder_params = optax.apply_updates(encoder_params, updates)

        targets = self._target(
            fixed_encoder_target_params, target_params, rewards, nxtobses, not_terminateds, key
        )
        params["values"]["min_value"] = jnp.minimum(jnp.min(targets), params["values"]["min_value"])
        params["values"]["max_value"] = jnp.maximum(jnp.max(targets), params["values"]["max_value"])

        (total_loss, (critic_loss, actor_loss, priority)), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, fixed_encoder_params, obses, actions, targets, key, step)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        target_params = hard_update(params, target_params, step, self.target_network_update_freq)
        fixed_encoder_target_params = hard_update(
            fixed_encoder_params, fixed_encoder_target_params, step, self.target_network_update_freq
        )
        fixed_encoder_params = hard_update(
            encoder_params, fixed_encoder_params, step, self.target_network_update_freq
        )
        return (
            encoder_params,
            params,
            fixed_encoder_params,
            fixed_encoder_target_params,
            target_params,
            encoder_opt_state,
            opt_state,
            repr_loss,
            critic_loss,
            -actor_loss,
            priority,
        )

    def _encoder_loss(self, encoder_params, obses, next_obses, actions, key):
        next_zs = jax.lax.stop_gradient(
            self.encoder(encoder_params, key, self.preproc(encoder_params, key, next_obses))
        )
        zs = self.encoder(encoder_params, key, self.preproc(encoder_params, key, obses))
        pred_zs = self.action_encoder(encoder_params, key, zs, actions)
        loss = jnp.mean(jnp.square(next_zs - pred_zs))
        return loss

    def _loss(self, params, fixed_encoder_params, obses, actions, targets, key, step):
        feature = self.preproc(fixed_encoder_params, key, obses)
        zs = self.encoder(fixed_encoder_params, key, feature)
        zsa = self.action_encoder(fixed_encoder_params, key, zs, actions)

        q1, q2 = self.critic(params, key, feature, zs, zsa, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(hubberloss(error1, 1.0)) + jnp.mean(hubberloss(error2, 1.0))

        priority = jnp.maximum(jnp.maximum(jnp.abs(error1), jnp.abs(error2)), 1.0)

        policy = self.actor(params, key, feature, zs)
        policy_zsa = self.action_encoder(fixed_encoder_params, key, zs, policy)
        Q1, Q2 = self.critic(jax.lax.stop_gradient(params), key, feature, zs, policy_zsa, policy)
        actor_loss = -jnp.mean(jnp.concatenate([Q1, Q2], axis=1))
        total_loss = jax.lax.select(
            step % self.policy_delay == 0, critic_loss + actor_loss, critic_loss
        )
        return total_loss, (critic_loss, actor_loss, priority)

    def _target(
        self, fixed_encoder_target_params, target_params, rewards, nxtobses, not_terminateds, key
    ):
        next_feature = self.preproc(fixed_encoder_target_params, key, nxtobses)
        fixed_target_zs = self.encoder(fixed_encoder_target_params, key, next_feature)

        next_action = jnp.clip(
            self.actor(target_params, key, next_feature, fixed_target_zs)
            + jnp.clip(
                self.target_action_noise
                * jax.random.normal(key, (self.batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )

        fixed_target_zsa = self.action_encoder(
            fixed_encoder_target_params, key, fixed_target_zs, next_action
        )

        q1, q2 = self.critic(
            target_params, key, next_feature, fixed_target_zs, fixed_target_zsa, next_action
        )
        next_q = jnp.clip(
            jnp.minimum(q1, q2),
            target_params["values"]["min_value"],
            target_params["values"]["max_value"],
        )
        return rewards + not_terminateds * self.gamma * next_q

    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(self.score_mean, self.loss_mean)

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="TD7",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        pbar = trange(total_timesteps, miniters=log_interval)
        self.eval_freq = total_timesteps // 100
        with TensorboardWriter(self.tensorboard_log, tb_log_name) as (
            self.summary,
            self.save_path,
        ):
            if self.env_type == "unity":
                score_mean = self.learn_unity(pbar, callback, log_interval)
            if self.env_type == "gym":
                score_mean = self.learn_gym(pbar, callback, log_interval)
            if self.env_type == "gymMultiworker":
                score_mean = self.learn_gymMultiworker(pbar, callback, log_interval)
            add_hparams(self, self.summary, {"env/episode_reward": score_mean}, total_timesteps)
            self.save_params(self.save_path)

    def learn_gym(self, pbar, callback=None, log_interval=100):
        state, info = self.env.reset()
        state = [np.expand_dims(state, axis=0)]
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size], dtype=np.int32)
        self.score_mean = None
        self.loss_mean = None
        for steps in pbar:
            self.eplen += 1
            actions = self.actions(state, steps)
            next_state, reward, terminated, truncated, info = self.env.step(actions[0])
            next_state = [np.expand_dims(next_state, axis=0)]
            self.replay_buffer.add(state, actions[0], reward, next_state, terminated, truncated)
            self.scores[0] += reward
            state = next_state
            if terminated or truncated:
                self.end_episode(steps, self.scores[0], self.eplen[0])
                self.scores[0] = 0
                self.eplen[0] = 0
                state, info = self.env.reset()
                state = [np.expand_dims(state, axis=0)]

            if (
                steps % log_interval == 0
                and self.loss_mean is not None
                and self.score_mean is not None
            ):
                pbar.set_description(self.discription())

            if steps % self.eval_freq == 0:
                self.score_mean = self.eval(steps)
        return self.eval(steps + 1)

    def eval(self, steps):
        total_reward = np.zeros(self.eval_eps)
        total_ep_len = np.zeros(self.eval_eps)
        total_truncated = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            state, info = self.eval_env.reset()
            state = [np.expand_dims(state, axis=0)]
            terminated = False
            truncated = False
            eplen = 0
            while not terminated and not truncated:
                actions = self.actions(
                    state, self.learning_starts + 1, use_checkpoint=True, exploration=False
                )
                next_state, reward, terminated, truncated, info = self.eval_env.step(actions[0])
                next_state = [np.expand_dims(next_state, axis=0)]
                # self.replay_buffer.add(state, actions[0], reward, next_state, terminated, truncated)
                total_reward[ep] += reward
                state = next_state
                eplen += 1
            total_ep_len[ep] = eplen
            total_truncated[ep] = float(truncated)

        mean_reward = np.mean(total_reward)

        if self.summary:
            self.summary.add_scalar("env/episode_reward", mean_reward, steps)
            self.summary.add_scalar("env/episode len", np.mean(total_ep_len), steps)
            self.summary.add_scalar("env/time over", np.mean(total_truncated), steps)
        return mean_reward
