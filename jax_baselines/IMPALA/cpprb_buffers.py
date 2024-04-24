import random
from collections import deque, namedtuple

import cpprb
import numpy as np
import ray
from ray.util.queue import Queue

batch = namedtuple(
    "batch_tuple",
    ["obses", "actions", "mu_log_prob", "rewards", "nxtobses", "terminateds", "truncateds"],
)


class EpochBuffer:
    def __init__(self, size: int, env_dict: dict):
        self.max_size = size
        self.env_dict = env_dict
        self.obsdict = dict((o, s) for o, s in env_dict.items() if o.startswith("obs"))
        self.nextobsdict = dict((o, s) for o, s in env_dict.items() if o.startswith("next_obs"))
        self.buffer = cpprb.ReplayBuffer(size, env_dict=env_dict)

    def __len__(self):
        return self.buffer.get_stored_size()

    def add(self, obs_t, action, log_prob, reward, nxtobs_t, terminated, truncted=False):
        obsdict = dict(zip(self.obsdict.keys(), [o for o in obs_t]))
        nextobsdict = dict(zip(self.nextobsdict.keys(), [no for no in nxtobs_t]))
        self.buffer.add(
            **obsdict,
            action=action,
            log_prob=log_prob,
            reward=reward,
            **nextobsdict,
            terminated=terminated,
            truncted=truncted
        )
        if terminated or terminated:
            self.buffer.on_episode_end()

    def get_buffer(self):
        trans = self.buffer.get_all_transitions()
        transitions = batch(
            [trans[o] for o in self.obsdict.keys()],
            trans["action"],
            trans["log_prob"],
            trans["reward"],
            [trans[o] for o in self.nextobsdict.keys()],
            trans["terminated"],
            trans["truncted"],
        )
        return transitions


@ray.remote(num_cpus=1)
class Buffer_getter:
    def __init__(self, queue, env_dict, actor_num, size, sample_size):
        self.queue = queue
        self.env_dict = env_dict
        self.actor_num = actor_num
        self.size = size
        self.replay = size > 0
        self.sample_size = sample_size
        if self.replay:
            self.replay_buffer = deque(maxlen=size)
            self._sample = self.replay_sample
        else:
            self._sample = self.queue_sample

    def sample(self):
        return self._sample()

    def queue_sample(self):
        gets = [self.queue.get() for idx in range(self.sample_size)]
        transitions = batch(*zip(*gets))
        return transitions

    def replay_sample(self):
        while True:
            self.replay_buffer.extend(self.queue.get_nowait_batch(self.queue.size()))
            if len(self.replay_buffer) >= self.sample_size:
                break
        gets = random.sample(self.replay_buffer, self.sample_size)
        transitions = batch(*zip(*gets))
        return transitions


class ImpalaBuffer:
    def __init__(
        self,
        replay_size: int,
        actor_num: int,
        observation_space: list,
        discrete=True,
        action_space=1,
        sample_size=32,
    ):
        self.max_size = replay_size
        self.actor_num = actor_num
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {"shape": o, "dtype": np.uint8}
                if len(o) >= 3
                else {"shape": o, "dtype": np.float32},
            )
            for idx, o in enumerate(observation_space)
        )
        self.nextobsdict = dict(
            (
                "next_obs{}".format(idx),
                {"shape": o, "dtype": np.uint8}
                if len(o) >= 3
                else {"shape": o, "dtype": np.float32},
            )
            for idx, o in enumerate(observation_space)
        )

        self.env_dict = {
            **self.obsdict,
            "action": {"shape": 1 if discrete else action_space},
            "log_prob": {},
            "reward": {},
            **self.nextobsdict,
            "terminated": {},
            "truncted": {},
        }

        self.queue = Queue(maxsize=max(actor_num * 2, replay_size))
        self.getter = Buffer_getter.remote(
            self.queue, self.env_dict, actor_num, replay_size, sample_size
        )
        self.get = self.getter.sample.remote()

    def queue_info(self):
        return self.queue, self.env_dict, self.actor_num

    def __len__(self):
        return len(self.queue)

    def queue_is_empty(self):
        return self.queue.empty()

    def sample(self):
        out = ray.get(self.get)
        self.get = self.getter.sample.remote()
        return out
