import gymnasium as gym
import os
import ray
import numpy as np

from abc import ABC, abstractmethod
from gymnasium import spaces
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.registry import default_registry
from jax_baselines.common.utils import convert_states

def get_env_builder(env_name, **kwargs):
    print(env_name)
    print(list(default_registry.keys()))
    if os.path.isfile(env_name) and os.path.exists(env_name):
        print("Unity ML-Agents")
        timescale = kwargs.get("timescale", 20)
        capture_frame_rate = kwargs.get("capture_frame_rate", 60)

        def env_builder(worker=None, worker_id=0):
            env = UnityMultiworker(env_name, False, worker_id, timescale=timescale, capture_frame_rate=capture_frame_rate)
            return env
        env_name = os.path.splitext(os.path.basename(env_name))[0]
        env_type = "unity"
        env_info = {
            "env_type": env_type,
            "env_id": env_name,
        }
    elif env_name in list(default_registry.keys()):
        print("Unity ML-Agents")
        timescale = kwargs.get("timescale", 20)
        capture_frame_rate = kwargs.get("capture_frame_rate", 60)
        def env_builder(worker=None, worker_id=0):
            env = UnityMultiworker(env_name, True, worker_id, timescale=timescale, capture_frame_rate=capture_frame_rate)
            return env
        env_name = os.path.splitext(os.path.basename(env_name))[0]
        env_type = "unity"
        env_info = {
            "env_type": env_type,
            "env_id": env_name,
        }
    else:
        def env_builder(worker=1):
            if worker > 1:
                return gymMultiworker(env_name, worker_num=worker)
            else:
                from jax_baselines.common.atari_wrappers import (
                    get_env_type,
                    make_wrap_atari,
                )

                env_type, env_id = get_env_type(env_name)
                if env_type == "atari_env":
                    env = make_wrap_atari(env_name, clip_rewards=True)
                else:
                    env = gym.make(env_name)
                return env
        env_type = "gym"
        env_info = {
            "env_type": env_type,
            "env_id": env_name,
        }
    return env_builder, env_info


class Multiworker(ABC):

    env_info = None
    num_workers = None

    @abstractmethod
    def __init__(self, env_id, worker_num=8):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def get_steps(self):
        pass

class UnityMultiworker(Multiworker):


    def __init__(self, env_id, in_registry=False, worker_id=0, **kwargs):

        if in_registry:
            self.env = default_registry[env_id].make(worker_id=worker_id)
        else:
            from mlagents_envs.environment import UnityEnvironment
            from mlagents_envs.side_channel.engine_configuration_channel import (
                EngineConfigurationChannel,
            )
            from mlagents_envs.side_channel.environment_parameters_channel import (
                EnvironmentParametersChannel,
            )
            engine_configuration_channel = EngineConfigurationChannel()
            channel = EnvironmentParametersChannel()
            
            timescale = kwargs.get("timescale", 20)
            capture_frame_rate = kwargs.get("capture_frame_rate", 60)
            engine_configuration_channel.set_configuration_parameters(
                time_scale=timescale, capture_frame_rate=capture_frame_rate
            )
            self.env = UnityEnvironment(
                file_name=env_id,
                worker_id=worker_id,
                no_graphics=False,
                side_channels=[engine_configuration_channel, channel],
                timeout_wait=10000,
            )

        self.env.reset()
        group_name = list(self.env.behavior_specs.keys())[0]
        group_spec = self.env.behavior_specs[group_name]
        
        self.env.step()
        dec, term = self.env.get_steps(group_name)
        self.group_name = group_name
        self.worker_num = len(dec.agent_id)

        observation_space = [list(spec.shape) for spec in group_spec.observation_specs]
        self.action_type = "discrete" if len(group_spec.action_spec.discrete_branches) > 0 else "continuous"
        action_size = [branch for branch in group_spec.action_spec.discrete_branches]
        self.env_info = {
            "observation_space": observation_space,
            "action_space": action_size,
            "env_type": "unity",
            "env_id": env_id,
        }

    def step(self, actions):
        action_tuple = ActionTuple(discrete=actions) if self.action_type == "discrete" else ActionTuple(continuous=actions)
        self.env.set_actions(self.group_name, action_tuple)
        self.env.step()

    def get_steps(self):
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
        states = convert_states(dec.obs)
        terminateds = np.full((self.worker_num), False)
        truncateds = np.full((self.worker_num), False)
        rewards = dec.reward
        return states, rewards, terminateds, truncateds, [dict() for _ in range(self.worker_num)], term_obses, term_ids

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
        states, end_states, rewards, terminateds, truncateds, infos = ray.get(self.steps)
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