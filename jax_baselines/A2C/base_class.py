from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from tqdm.auto import trange

from jax_baselines.A2C.transition_buffer import EpochBuffer
from jax_baselines.common.base_classes import (
    TensorboardWriter,
    restore,
    save,
    select_optimizer,
)
from jax_baselines.common.utils import add_hparams, convert_jax, convert_states, key_gen
from jax_baselines.common.worker import gymMultiworker


class Actor_Critic_Policy_Gradient_Family(object):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=32,
        val_coef=0.2,
        ent_coef=0.01,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        self.name = "Actor_Critic_Policy_Gradient_Family"
        self.env = env
        self.model_builder_maker = model_builder_maker
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.hidden_space = []

        self.params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate)

        self.get_env_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = restore(path)

    def get_memory_setup(self):
        self.buffer = EpochBuffer(
            self.batch_size,
            self.observation_space,
            self.hidden_space,
            self.worker_size,
            [1] if self.action_type == "discrete" else self.action_size,
        )

    def get_env_setup(self):
        print("----------------------env------------------------")
        if isinstance(self.env, UnityEnvironment):
            print("unity-ml agent environmet")
            self.env.reset()
            group_name = list(self.env.behavior_specs.keys())[0]
            group_spec = self.env.behavior_specs[group_name]
            self.env.step()
            dec, term = self.env.get_steps(group_name)
            self.group_name = group_name

            self.observation_space = [list(spec.shape) for spec in group_spec.observation_specs]
            if group_spec.action_spec.continuous_size == 0:
                self.action_size = [branch for branch in group_spec.action_spec.discrete_branches]
                self.action_type = "discrete"
                self.conv_action = lambda a: ActionTuple(discrete=a)
            else:
                self.action_size = [group_spec.action_spec.continuous_size]
                self.action_type = "continuous"
                self.conv_action = lambda a: ActionTuple(
                    continuous=np.clip(a, -3.0, 3.0) / 3.0
                )  # np.clip(a, -3.0, 3.0) / 3.0)
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"

        elif isinstance(self.env, gym.Env) or isinstance(self.env, gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            if not isinstance(action_space, spaces.Box):
                self.action_size = [action_space.n]
                self.action_type = "discrete"
                self.conv_action = lambda a: a[0][0]
            else:
                self.action_size = [action_space.shape[0]]
                self.action_type = "continuous"
                self.conv_action = lambda a: np.clip(a[0], -3.0, 3.0) / 3.0
            self.worker_size = 1
            self.env_type = "gym"

        elif isinstance(self.env, gymMultiworker):
            print("gymMultiworker")
            env_info = self.env.env_info
            self.observation_space = [list(env_info["observation_space"].shape)]
            if not isinstance(env_info["action_space"], spaces.Box):
                self.action_size = [env_info["action_space"].n]
                self.action_type = "discrete"
                self.conv_action = lambda a: a
            else:
                self.action_size = [env_info["action_space"].shape[0]]
                self.action_type = "continuous"
                self.conv_action = lambda a: np.clip(a, -3.0, 3.0) / 3.0
            self.worker_size = self.env.worker_num
            self.env_type = "gymMultiworker"

        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        if self.action_type == "discrete":
            self._get_actions = self._get_actions_discrete
            self.get_logprob = self.get_logprob_discrete
            self._loss = self._loss_discrete
            self.actions = self.action_discrete
        elif self.action_type == "continuous":
            self._get_actions = self._get_actions_continuous
            self.get_logprob = self.get_logprob_continuous
            self._loss = self._loss_continuous
            self.actions = self.action_continuous

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions_discrete(self, params, obses, key=None) -> jnp.ndarray:
        prob = jax.nn.softmax(
            self.actor(params, key, self.preproc(params, key, convert_jax(obses))),
            axis=1,
        )
        return prob

    def _get_actions_continuous(self, params, obses, key=None) -> jnp.ndarray:
        mu, std = self.actor(params, key, self.preproc(params, key, convert_jax(obses)))
        return mu, jnp.exp(std)

    def action_discrete(self, obs):
        prob = np.asarray(self._get_actions(self.params, obs))
        return np.expand_dims(
            np.stack([np.random.choice(self.action_size[0], p=p) for p in prob], axis=0),
            axis=1,
        )

    def action_continuous(self, obs):
        mu, std = self._get_actions(self.params, obs)
        return np.random.normal(mu, std)

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = jnp.clip(jax.nn.softmax(prob), 1e-5, 1.0)
        action = action.astype(jnp.int32)
        if out_prob:
            return prob, jnp.log(jnp.take_along_axis(prob, action, axis=1))
        else:
            return jnp.log(jnp.take_along_axis(prob, action, axis=1))

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        mu, log_std = prob
        std = jnp.exp(log_std)
        if out_prob:
            return prob, -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )
        else:
            return -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )

    def _loss_continuous(self):
        pass

    def _loss_discrete(self):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        tb_log_name="Q_network",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        pbar = trange(total_timesteps, miniters=log_interval, smoothing=0.01)
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

    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
            np.mean(self.scoreque), np.mean(self.lossque)
        )

    def learn_unity(self, pbar, callback=None, log_interval=100):
        self.env.reset()
        self.env.step()
        dec, term = self.env.get_steps(self.group_name)
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        obses = convert_states(dec.obs)
        for steps in pbar:
            self.eplen += 1
            actions = self.actions(obses)
            action_tuple = self.conv_action(actions)

            self.env.set_actions(self.group_name, action_tuple)
            self.env.step()

            dec, term = self.env.get_steps(self.group_name)
            term_ids = term.agent_id
            term_obses = convert_states(term.obs)
            term_rewards = term.reward
            term_done = term.interrupted
            while len(dec) == 0:
                self.env.step()
                dec, term = self.env.get_steps(self.group_name)
                if len(term.agent_id):
                    term_ids = np.append(term_ids, term.agent_id)
                    term_obses = [
                        np.concatenate((to, o), axis=0)
                        for to, o in zip(term_obses, convert_states(term.obs))
                    ]
                    term_rewards = np.append(term_rewards, term.reward)
                    term_done = np.append(term_done, term.interrupted)
            nxtobs = convert_states(dec.obs)
            terminated = np.full((self.worker_size), False)
            terminated = np.full((self.worker_size), False)
            reward = dec.reward
            term_on = len(term_ids) > 0
            if term_on:
                nxtobs_t = [n.at[term_ids].set(t) for n, t in zip(nxtobs, term_obses)]
                terminated[term_ids] = np.logical_not(term_done)
                terminated[term_ids] = True
                reward[term_ids] = term_rewards
                self.buffer.add(
                    obses,
                    actions,
                    np.expand_dims(reward, axis=1),
                    nxtobs_t,
                    np.expand_dims(terminated, axis=1),
                    np.expand_dims(terminated, axis=1),
                )
            else:
                self.buffer.add(
                    obses,
                    actions,
                    np.expand_dims(reward, axis=1),
                    nxtobs,
                    np.expand_dims(terminated, axis=1),
                    np.expand_dims(terminated, axis=1),
                )
            self.scores += reward
            obses = nxtobs
            if term_on:
                if self.summary:
                    self.summary.add_scalar(
                        "env/episode_reward", np.mean(self.scores[term_ids]), steps
                    )
                    self.summary.add_scalar("env/episode len", np.mean(self.eplen[term_ids]), steps)
                    self.summary.add_scalar(
                        "env/time over",
                        np.mean(1 - terminated[term_ids].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[term_ids])
                self.scores[term_ids] = reward[term_ids]
                self.eplen[term_ids] = 0

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
        return np.mean(self.scoreque)

    def learn_gym(self, pbar, callback=None, log_interval=100):
        state, info = self.env.reset()
        have_original_reward = "original_reward" in info.keys()
        have_lives = "lives" in info.keys()
        state = [np.expand_dims(state, axis=0)]
        if have_original_reward:
            self.original_score = np.zeros([self.worker_size])
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            actions = self.actions(state)
            next_state, reward, terminated, truncated, info = self.env.step(self.conv_action(actions))
            next_state = [np.expand_dims(next_state, axis=0)]
            self.buffer.add(state, actions[0], reward, next_state, terminated, truncated)
            if have_original_reward:
                self.original_score[0] += info["original_reward"]
            self.scores[0] += reward
            state = next_state
            if terminated or truncated:
                self.scoreque.append(self.scores[0])
                if self.summary:
                    if have_original_reward:
                        if have_lives:
                            if info["lives"] == 0:
                                self.summary.add_scalar(
                                    "env/original_reward", self.original_score[0], steps
                                )
                                self.original_score[0] = 0
                        else:
                            self.summary.add_scalar(
                                "env/original_reward", self.original_score[0], steps
                            )
                            self.original_score[0] = 0
                    self.summary.add_scalar("env/episode_reward", self.scores[0], steps)
                    self.summary.add_scalar("env/episode len", self.eplen[0], steps)
                    self.summary.add_scalar("env/time over", float(truncated), steps)
                self.scores[0] = 0
                self.eplen[0] = 0
                state, info = self.env.reset()
                state = [np.expand_dims(state, axis=0)]

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
        return np.mean(self.scoreque)

    def learn_gymMultiworker(self, pbar, callback=None, log_interval=100):
        state, _, _, _, info, _, _ = self.env.get_steps()
        have_original_reward = "original_reward" in info[0].keys()
        have_lives = "lives" in info[0].keys()
        if have_original_reward:
            self.original_score = np.zeros([self.worker_size])
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            actions = self.actions([state])
            self.env.step(self.conv_action(actions))

            (
                real_nextstates,
                rewards,
                terminateds,
                truncated,
                infos,
                end_states,
                end_idx,
            ) = self.env.get_steps()
            self.scores += rewards
            if have_original_reward:
                self.original_score += np.asarray([info["original_reward"] for info in infos])
            if end_states is not None:
                nxtstates = np.copy(real_nextstates)
                nxtstates[end_idx] = end_states
                if self.summary:
                    if have_original_reward:
                        if have_lives:
                            end_lives = np.asarray([infos[ei]["lives"] for ei in end_idx])
                            done_lives = np.logical_not(end_lives)
                            if np.sum(done_lives) > 0:
                                self.summary.add_scalar(
                                    "env/original_reward",
                                    np.mean(self.original_score[end_idx[done_lives]]),
                                    steps,
                                )
                                self.original_score[end_idx[done_lives]] = 0
                        else:
                            self.summary.add_scalar(
                                "env/original_reward", self.original_score[end_idx], steps
                            )
                            self.original_score[end_idx] = 0
                    self.summary.add_scalar(
                        "env/episode_reward", np.mean(self.scores[end_idx]), steps
                    )
                    self.summary.add_scalar("env/episode len", np.mean(self.eplen[end_idx]), steps)
                    self.summary.add_scalar(
                        "env/time over",
                        np.mean(1 - terminateds[end_idx].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[end_idx])
                self.scores[end_idx] = 0
                self.eplen[end_idx] = 0
                self.buffer.add(
                    [state],
                    actions,
                    np.expand_dims(rewards, axis=1),
                    [nxtstates],
                    np.expand_dims(terminateds, axis=1),
                    np.expand_dims(truncated, axis=1),
                )
            else:
                self.buffer.add(
                    [state],
                    actions,
                    np.expand_dims(rewards, axis=1),
                    [real_nextstates],
                    np.expand_dims(terminateds, axis=1),
                    np.expand_dims(truncated, axis=1),
                )
            state = real_nextstates

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
        return np.mean(self.scoreque)

    def test(self, episode=10, tb_log_name=None):
        if tb_log_name is None:
            tb_log_name = self.save_path

        directory = tb_log_name
        if self.env_type == "gym":
            self.test_gym(episode, directory)
        if self.env_type == "gymMultiworker":
            self.test_gymMultiworker(episode, directory)

    def test_unity(self, episode, directory):
        pass

    def test_gymMultiworker(self, episode, directory):
        from gymnasium.wrappers import RecordVideo

        env_id = self.env.env_id
        from jax_baselines.common.atari_wrappers import get_env_type, make_wrap_atari

        env_type, env_id = get_env_type(env_id)
        if env_type == "atari_env":
            env = make_wrap_atari(env_id, clip_rewards=True)
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        Render_env = RecordVideo(env, directory, episode_trigger=lambda x: True)
        for i in range(episode):
            state, info = Render_env.reset()
            state = [np.expand_dims(state, axis=0)]
            terminated = False
            truncated = False
            episode_rew = 0
            while not (terminated or truncated):
                actions = self.actions(state)
                observation, reward, terminated, truncated, info = Render_env.step(
                    actions[0][0] if self.action_type == "discrete" else actions[0]
                )
                state = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)

    def test_gym(self, episode, directory):
        from gymnasium.wrappers import RecordVideo

        Render_env = RecordVideo(self.env, directory, episode_trigger=lambda x: True)
        for i in range(episode):
            state, info = Render_env.reset()
            state = [np.expand_dims(state, axis=0)]
            terminated = False
            truncated = False
            episode_rew = 0
            while not (terminated or truncated):
                actions = self.actions(state)
                observation, reward, terminated, truncated, info = Render_env.step(
                    actions[0][0] if self.action_type == "discrete" else actions[0]
                )
                state = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)
