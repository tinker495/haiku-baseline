import cpprb
import numpy as np


class EpochBuffer(object):
    def __init__(self, epoch_size: int, observation_space: list, worker_size=1, action_space=1):
        self.epoch_size = epoch_size
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
        self.obscompress = None
        self.worker_size = worker_size
        self.local_buffers = [
            cpprb.ReplayBuffer(
                epoch_size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                    "truncated": {},
                },
            )
            for _ in range(worker_size)
        ]

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated):
        for w in range(self.worker_size):
            obsdict = dict(zip(self.obsdict.keys(), [o[w] for o in obs_t]))
            nextobsdict = dict(zip(self.nextobsdict.keys(), [no[w] for no in nxtobs_t]))
            self.local_buffers[w].add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                terminated=terminated[w],
                truncated=truncated[w],
            )
            if terminated[w] or truncated[w]:
                self.local_buffers[w].on_episode_end()

    def get_buffer(self):
        transitions = {
            "obses": [],
            "actions": [],
            "rewards": [],
            "nxtobses": [],
            "terminateds": [],
            "truncateds": [],
        }
        for w in range(self.worker_size):
            trans = self.local_buffers[w].get_all_transitions()
            transitions["obses"].append([trans[o] for o in self.obsdict.keys()])
            transitions["actions"].append(trans["action"])
            transitions["rewards"].append(trans["reward"])
            transitions["nxtobses"].append([trans[o] for o in self.nextobsdict.keys()])
            transitions["terminateds"].append(trans["terminated"])
            transitions["truncateds"].append(trans["truncated"])
            self.local_buffers[w].clear()
        return transitions


class ReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        compress_memory=False,
        env_dict=None,
        n_s=None,
    ):
        self.max_size = size
        if env_dict is None:
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
            self.obscompress = None
            if compress_memory:
                self.obscompress = []
                for k in self.obsdict:
                    if len(self.obsdict[k]["shape"]) >= 3:
                        self.obscompress.append(k)
                        del self.nextobsdict[f"next_{k}"]

            self.buffer = cpprb.ReplayBuffer(
                size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                },
                next_of=self.obscompress,
                stack_compress=self.obscompress,
            )
        else:
            self.obsdict = dict((o, None) for o in env_dict.keys() if o.startswith("obs"))
            self.nextobsdict = dict((o, None) for o in env_dict.keys() if o.startswith("next_obs"))
            self.buffer = cpprb.ReplayBuffer(size, env_dict=env_dict, Nstep=n_s)

    def __len__(self) -> int:
        return self.buffer.get_stored_size()

    @property
    def storage(self):
        return self.buffer

    @property
    def buffer_size(self) -> int:
        return self.max_size

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.max_size

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        obsdict = dict(zip(self.obsdict.keys(), obs_t))
        nextobsdict = dict(zip(self.nextobsdict.keys(), nxtobs_t))
        self.buffer.add(**obsdict, action=action, reward=reward, **nextobsdict, terminated=terminated)

    def episode_end(self):
        self.buffer.on_episode_end()

    def sample(self, batch_size: int):
        smpl = self.buffer.sample(batch_size)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["terminated"],
        }

    def get_buffer(self):
        return self.buffer.get_all_transitions()

    def conv_transitions(self, transitions):
        return {
            "obses": [transitions[o] for o in self.obsdict.keys()],
            "actions": transitions["action"],
            "rewards": transitions["reward"],
            "nxtobses": [transitions[no] for no in self.nextobsdict.keys()],
            "terminateds": transitions["terminated"],
        }

    def clear(self):
        self.buffer.clear()


class NstepReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        action_space=1,
        worker_size=1,
        n_step=1,
        gamma=0.99,
        compress_memory=False,
    ):
        self.max_size = size
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
        self.obscompress = None
        if compress_memory:
            self.obscompress = []
            for k in self.obsdict:
                if len(self.obsdict[k]["shape"]) >= 3:
                    self.obscompress.append(k)
                    del self.nextobsdict[f"next_{k}"]

        self.worker_size = worker_size
        n_s = {
            "size": n_step,
            "rew": "reward",
            "gamma": gamma,
            "next": list(self.nextobsdict.keys()),
        }

        if worker_size > 1:
            self.buffer = cpprb.ReplayBuffer(
                size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                },
                next_of=self.obscompress,
                stack_compress=self.obscompress,
            )
            self.local_buffers = [
                cpprb.ReplayBuffer(
                    2000,
                    env_dict={
                        **self.obsdict,
                        "action": {"shape": action_space},
                        "reward": {},
                        **self.nextobsdict,
                        "terminated": {},
                    },
                    Nstep=n_s,
                )
                for _ in range(worker_size)
            ]
            self.add = self.multiworker_add
        else:
            self.buffer = cpprb.ReplayBuffer(
                size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                },
                Nstep=n_s,
                next_of=self.obscompress,
                stack_compress=self.obscompress,
            )

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        super().add(obs_t, action, reward, nxtobs_t, terminated, truncated)
        if terminated or truncated:
            self.buffer.on_episode_end()

    def multiworker_add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        for w in range(self.worker_size):
            obsdict = dict(zip(self.obsdict.keys(), [o[w] for o in obs_t]))
            nextobsdict = dict(zip(self.nextobsdict.keys(), [no[w] for no in nxtobs_t]))
            self.local_buffers[w].add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                terminated=terminated[w],
            )
            if terminated[w] or truncated[w]:
                self.local_buffers[w].on_episode_end()
                self.buffer.add(**self.local_buffers[w].get_all_transitions())
                self.local_buffers[w].clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        alpha: float,
        action_space=1,
        compress_memory=False,
        eps=1e-4,
    ):
        self.max_size = size
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
        self.obscompress = None
        if compress_memory:
            self.obscompress = []
            for k in self.obsdict:
                if len(self.obsdict[k]["shape"]) >= 3:
                    self.obscompress.append(k)
                    del self.nextobsdict[f"next_{k}"]

        self.buffer = cpprb.PrioritizedReplayBuffer(
            size,
            env_dict={
                **self.obsdict,
                "action": {"shape": action_space},
                "reward": {},
                **self.nextobsdict,
                "terminated": {},
            },
            alpha=alpha,
            eps=eps,
            next_of=self.obscompress,
            stack_compress=self.obscompress,
        )

    def sample(self, batch_size: int, beta=0.5):
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["terminated"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)


class PrioritizedNstepReplayBuffer(NstepReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        action_space=1,
        worker_size=1,
        n_step=1,
        gamma=0.99,
        alpha=0.4,
        compress_memory=False,
        eps=1e-4,
    ):
        self.max_size = size
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
        self.obscompress = None
        if compress_memory:
            self.obscompress = []
            for k in self.obsdict:
                if len(self.obsdict[k]["shape"]) >= 3:
                    self.obscompress.append(k)
                    del self.nextobsdict[f"next_{k}"]

        self.worker_size = worker_size
        n_s = {
            "size": n_step,
            "rew": "reward",
            "gamma": gamma,
            "next": list(self.nextobsdict.keys()),
        }

        if worker_size > 1:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                },
                alpha=alpha,
                eps=eps,
                next_of=self.obscompress,
                stack_compress=self.obscompress,
            )
            self.local_buffers = [
                cpprb.ReplayBuffer(
                    2000,
                    env_dict={
                        **self.obsdict,
                        "action": {"shape": action_space},
                        "reward": {},
                        **self.nextobsdict,
                        "terminated": {},
                    },
                    Nstep=n_s,
                )
                for _ in range(worker_size)
            ]
            self.add = self.multiworker_add
        else:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                },
                alpha=alpha,
                eps=eps,
                Nstep=n_s,
                next_of=self.obscompress,
                stack_compress=self.obscompress,
            )

    def sample(self, batch_size: int, beta=0.5):
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["terminated"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)


class MultiPrioritizedReplayBuffer:
    def __init__(
        self,
        size: int,
        observation_space: list,
        alpha: float,
        action_space=1,
        n_step=1,
        gamma=0.99,
        manager=None,
        compress_memory=False,
        eps=1e-4,
    ):
        self.max_size = size
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
        self.obscompress = None
        if compress_memory:
            self.obscompress = []
            for k in self.obsdict:
                if len(self.obsdict[k]["shape"]) >= 3:
                    self.obscompress.append(k)
                    del self.nextobsdict[f"next_{k}"]

        self.env_dict = {
            **self.obsdict,
            "action": {"shape": action_space},
            "reward": {},
            **self.nextobsdict,
            "terminated": {},
        }

        self.n_s = None
        if n_step > 1:
            self.n_s = {
                "size": n_step,
                "rew": "reward",
                "gamma": gamma,
                "next": list(self.nextobsdict.keys()),
            }

        self.buffer = cpprb.MPPrioritizedReplayBuffer(
            size,
            env_dict=self.env_dict,
            alpha=alpha,
            eps=eps,
            ctx=manager,
            backend="SharedMemory",
        )

    def __len__(self):
        return self.buffer.get_stored_size()

    def buffer_info(self):
        return self.buffer, self.env_dict, self.n_s

    def sample(self, batch_size: int, beta=0.5):
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["terminated"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)
