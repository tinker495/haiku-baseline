from collections import deque

import gymnasium as gym
import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from tqdm.auto import trange

from jax_baselines.common.base_classes import (
    TensorboardWriter,
    restore,
    save,
    select_optimizer,
)
from jax_baselines.common.cpprb_buffers import (
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from jax_baselines.common.schedules import ConstantSchedule, LinearSchedule
from jax_baselines.common.utils import add_hparams, convert_states, key_gen
from jax_baselines.common.worker import gymMultiworker


class Q_Network_Family(object):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        double_q=False,
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
        self.name = "Q_Network_Family"
        self.env = env
        self.model_builder_maker = model_builder_maker
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = int(
            np.ceil(target_network_update_freq / train_freq) * train_freq
        )
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._gamma = np.power(gamma, n_step)  # n_step gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.dueling_model = dueling_model
        self.n_step_method = n_step > 1
        self.n_step = n_step
        self.munchausen = munchausen
        self.munchausen_alpha = 0.9
        self.munchausen_entropy_tau = 0.03

        self.train_steps_count = 0
        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-2 / self.batch_size)

        self.compress_memory = compress_memory

        self.get_env_setup()
        self.get_memory_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

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
            self.action_size = [branch for branch in group_spec.action_spec.discrete_branches]
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"

        elif isinstance(self.env, gym.Env) or isinstance(self.env, gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.n]
            self.worker_size = 1
            self.env_type = "gym"

        elif isinstance(self.env, gymMultiworker):
            print("gymMultiworker")
            env_info = self.env.env_info
            self.observation_space = [list(env_info["observation_space"].shape)]
            self.action_size = [env_info["action_space"].n]
            self.worker_size = self.env.worker_num
            self.env_type = "gymMultiworker"

        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")

    def get_memory_setup(self):
        if self.prioritized_replay:
            if self.n_step_method:
                self.replay_buffer = PrioritizedNstepReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    1,
                    self.worker_size,
                    self.n_step,
                    self.gamma,
                    self.prioritized_replay_alpha,
                    False,
                    self.prioritized_replay_eps,
                )
            else:
                self.replay_buffer = PrioritizedReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.prioritized_replay_alpha,
                    1,
                    False,
                    self.prioritized_replay_eps,
                )

        else:
            if self.n_step_method:
                self.replay_buffer = NstepReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    1,
                    self.worker_size,
                    self.n_step,
                    self.gamma,
                )
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space, 1)

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def actions(self, obs, epsilon):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    self.params, obs, next(self.key_seq) if self.param_noise else None
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def discription(self):
        if self.param_noise:
            return "score : {:.3f}, loss : {:.3f} |".format(
                np.mean(self.scoreque), np.mean(self.lossque)
            )
        else:
            return "score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                np.mean(self.scoreque), self.update_eps, np.mean(self.lossque)
            )

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
            tb_log_name = "{}Step_".format(self.n_step) + tb_log_name
        if self.prioritized_replay:
            tb_log_name = tb_log_name + "+PER"
        return tb_log_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        tb_log_name="Q_network",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        tb_log_name = self.tb_log_name_update(tb_log_name)

        if self.param_noise:
            self.exploration = ConstantSchedule(0)
        else:
            self.exploration = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                initial_p=self.exploration_initial_eps,
                final_p=self.exploration_final_eps,
            )
        self.update_eps = 1.0

        pbar = trange(total_timesteps, miniters=log_interval)
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
            actions = self.actions(obses, self.update_eps)
            action_tuple = ActionTuple(discrete=actions)

            self.env.set_actions(self.group_name, action_tuple)
            self.env.step()

            if (
                steps > self.learning_starts and steps % self.train_freq == 0
            ):  # train in step the environments
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)

            dec, term = self.env.get_steps(self.group_name)
            term_ids = term.agent_id
            term_obses = convert_states(term.obs)
            term_rewards = term.reward
            term_interrupted = term.interrupted
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
                    term_interrupted = np.append(term_interrupted, term.interrupted)
            nxtobs = convert_states(dec.obs)
            terminated = np.full((self.worker_size), False)
            truncated = np.full((self.worker_size), False)
            reward = dec.reward
            term_on = len(term_ids) > 0
            if term_on:
                nxtobs_t = [n.at[term_ids].set(t) for n, t in zip(nxtobs, term_obses)]
                terminated[term_ids] = term_interrupted
                truncated[term_ids] = np.logical_not(term_interrupted)
                reward[term_ids] = term_rewards
                self.replay_buffer.add(obses, actions, reward, nxtobs_t, terminated, truncated)
            else:
                self.replay_buffer.add(obses, actions, reward, nxtobs, terminated, truncated)
            self.scores += reward
            obses = nxtobs
            if term_on:
                if self.summary:
                    self.summary.add_scalar(
                        "env/episode_reward", np.mean(self.scores[term_ids]), steps
                    )
                    self.summary.add_scalar("env/episode_len", np.mean(self.eplen[term_ids]), steps)
                    self.summary.add_scalar(
                        "env/time_over",
                        np.mean(truncated[term_ids].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[term_ids])
                self.scores[term_ids] = reward[term_ids]
                self.eplen[term_ids] = 0

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
            actions = self.actions(state, self.update_eps)
            next_state, reward, terminated, truncated, info = self.env.step(actions[0][0])
            next_state = [np.expand_dims(next_state, axis=0)]
            self.replay_buffer.add(state, actions[0], reward, next_state, terminated, truncated)
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
                    self.summary.add_scalar("env/episode_len", self.eplen[0], steps)
                    self.summary.add_scalar("env/time_over", float(truncated), steps)
                self.scores[0] = 0
                self.eplen[0] = 0
                state, info = self.env.reset()
                state = [np.expand_dims(state, axis=0)]

            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps, self.gradient_steps)
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
            self.update_eps = self.exploration.value(steps)
            actions = self.actions([state], self.update_eps)
            self.env.step(actions)

            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)

            (
                real_nextstates,
                rewards,
                terminateds,
                truncateds,
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
                    self.summary.add_scalar("env/episode_len", np.mean(self.eplen[end_idx]), steps)
                    self.summary.add_scalar(
                        "env/time_over",
                        np.mean(truncateds[end_idx].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[end_idx])
                self.scores[end_idx] = 0
                self.eplen[end_idx] = 0
                self.replay_buffer.add([state], actions, rewards, [nxtstates], terminateds, truncateds)
            else:
                self.replay_buffer.add(
                    [state], actions, rewards, [real_nextstates], terminateds, truncateds
                )
            state = real_nextstates
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
        return np.mean(self.scoreque)

    def test(self, episode=10, tb_log_name=None):
        if tb_log_name is None:
            tb_log_name = self.save_path

        directory = tb_log_name
        if self.env_type == "gym":
            self.test_gym(episode, directory)

    def test_unity(self, episode, directory):
        pass

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
                actions = self.actions(state, 0.001)
                observation, reward, terminated, truncated, info = Render_env.step(actions[0][0])
                state = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
            Render_env.close()
            print("episod reward :", episode_rew)
