from abc import ABC

import gymnasium as gym
import numpy as np
import ray
from gymnasium import spaces


class Multiworker(ABC):
    def __init__(self, env_id, worker_num=8):
        pass

    def step(self, actions):
        pass

    def get_steps(self):
        pass


class gymMultiworker(Multiworker):
    def __init__(self, env_id, worker_num=8, render=False):
        ray.init(num_cpus=worker_num)
        self.env_id = env_id
        self.worker_num = worker_num
        self.workers = [
            gymRayworker.remote(env_id, render=(w == 0) if render else False)
            for w in range(worker_num)
        ]
        self.env_info = ray.get(self.workers[0].get_info.remote())
        self.steps = [w.get_reset.remote() for w in self.workers]

    def step(self, actions):
        self.steps = [w.step.remote(a) for w, a in zip(self.workers, actions)]

    def get_steps(self):
        states = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        end_states = []
        end_idx = []
        for idx, (state, end_state, reward, terminated, truncated, info) in enumerate(ray.get(self.steps)):
            states.append(state)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            if end_state is not None:
                end_states.append(end_state)
                end_idx.append(idx)
        states = np.stack(states, axis=0)
        rewards = np.stack(rewards, axis=0)
        terminateds = np.stack(terminateds, axis=0)
        truncateds = np.stack(truncateds, axis=0)
        if len(end_states):
            end_states = np.stack(end_states, axis=0)
            end_idx = np.stack(end_idx, axis=0)
        else:
            end_states = None
            end_idx = None
        return states, rewards, terminateds, truncateds, infos, end_states, end_idx

    def close(self):
        ray.shutdown()


@ray.remote
class gymRayworker:
    def __init__(self, env_name_, render=False):
        from jax_baselines.common.atari_wrappers import get_env_type, make_wrap_atari

        self.env_type, self.env_id = get_env_type(env_name_)
        if self.env_type == "atari_env":
            self.env = make_wrap_atari(env_name_, clip_rewards=True)
        else:
            self.env = gym.make(env_name_)
        if not isinstance(self.env.action_space, spaces.Box):
            self.action_conv = lambda a: a[0]
        else:
            self.action_conv = lambda a: a
        self.render = render

    def get_reset(self):
        state, info = self.env.reset()
        return state, None, 0, False, False, info

    def get_info(self):
        return {
            "observation_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "env_type": self.env_type,
            "env_id": self.env_id,
        }

    def step(self, action):
        if self.render:
            self.env.render()
        state, reward, terminated, truncated, info = self.env.step(self.action_conv(action))
        if terminated or truncated:
            end_state = state
            state, _ = self.env.reset()
        else:
            end_state = None
        return state, end_state, reward, terminated, truncated, info
