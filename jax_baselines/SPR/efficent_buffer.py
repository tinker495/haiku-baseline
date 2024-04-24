import random

import numpy as np


class Buffer(object):
    def __init__(self, size: int, obs_dict: dict, env_dict: dict):
        self.max_size = size
        self._idx = -1
        self.ep_idx = 0
        self.obs_dict = obs_dict
        self.env_dict = env_dict
        self.buffer = self.creat_buffer(size, obs_dict, env_dict)

    def creat_buffer(self, size: int, obs_dict: dict, env_dict: dict):
        buffer = {}
        for name, data in obs_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        for name, data in env_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        buffer["terminated"] = np.ones((size, 1), dtype=np.bool_)
        buffer["ep_idx"] = np.ones((size, 1), dtype=np.int32) * -1
        return buffer

    def get_stored_size(self):
        return min(self._idx, self.max_size)

    def update_idx(self):
        self._idx = self._idx + 1

    def add(self, obs, next_obs, **kwargs):
        self.update_idx()
        if self.buffer["ep_idx"][self.roll_idx_m1] != self.ep_idx:
            for idx, k in enumerate(self.obs_dict.keys()):
                self.buffer[k][self.roll_idx] = obs[idx]
        for idx, k in enumerate(self.obs_dict.keys()):
            self.buffer[k][self.next_roll_idx] = next_obs[idx]
        for k, data in kwargs.items():
            self.buffer[k][self.roll_idx] = data
        self.buffer["ep_idx"][self.roll_idx] = self.ep_idx
        return self.roll_idx

    def on_episode_end(self, truncated):
        if truncated:
            self.update_idx()
            self.buffer["ep_idx"][self.roll_idx] = -1
        self.ep_idx += 1

    def sample(self, idxs, traj_len=5):
        is_last = np.equal(self.buffer["ep_idx"][idxs], -1)[..., 0]
        idxs = np.where(is_last, idxs - 1, idxs)
        idxs = np.expand_dims(idxs, axis=1)
        obs_traj_idxs = (
            idxs + np.reshape(np.arange(traj_len + 1), (1, traj_len + 1))
        ) % self.max_size
        traj_idxs = (idxs + np.reshape(np.arange(traj_len), (1, traj_len))) % self.max_size
        obs = []
        for k in self.obs_dict:
            obs.append(self.buffer[k][obs_traj_idxs])
        data = {}
        for k in self.env_dict:
            data[k] = self.buffer[k][traj_idxs]
        terminated = self.buffer["terminated"][traj_idxs, 0]
        filled = np.equal(self.buffer["ep_idx"][idxs], self.buffer["ep_idx"][traj_idxs])[..., 0]
        filled = np.logical_and(filled, np.logical_not(terminated))
        return obs, data, terminated, filled, idxs

    @property
    def roll_idx_m1(self):
        return (self._idx - 1) % self.max_size

    @property
    def next_roll_idx(self):
        return (self._idx + 1) % self.max_size

    @property
    def roll_idx(self):
        return self._idx % self.max_size


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=np.int32)
        self.n_entries = 0
        self.max_priority = 1.0
        self.min_priority = np.inf

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # all sum of priorities
        return self.tree[0]

    def max(self):
        # return the max priority
        return self.max_priority

    def min(self):
        # return the min priority
        return self.min_priority

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p, minmax_decay=1e-4):
        self.max_priority = max(p, self.max_priority * (1.0 - minmax_decay))
        self.min_priority = min(p, self.min_priority * (1.0 + minmax_decay))
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class TransitionReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        prediction_depth=5,
    ):
        self.max_size = size
        self.prediction_depth = prediction_depth
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {
                    "shape": o,
                    "dtype": np.uint8 if len(o) >= 3 else np.float32,
                },
            )
            for idx, o in enumerate(observation_space)
        )

        if isinstance(action_space, int):
            action_space = (action_space,)

        self.buffer = Buffer(
            size,
            obs_dict=self.obsdict,
            env_dict={
                "actions": {"shape": action_space, "dtype": np.float32},
                "rewards": {"shape": (), "dtype": np.float32},
            },
        )

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
        self.buffer.add(
            obs_t,
            nxtobs_t,
            actions=action,
            rewards=reward,
            terminated=terminated,
        )
        if terminated or truncated:
            self.buffer.on_episode_end(truncated)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.buffer.get_stored_size(), size=batch_size)
        obs, data, terminated, filled, _ = self.buffer.sample(idxs, traj_len=self.prediction_depth)
        return {"obses": obs, **data, "terminateds": terminated, "filled": filled}


class PrioritizedTransitionReplayBuffer(TransitionReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        prediction_depth=5,
        alpha: float = 0.6,
        eps: float = 1e-4,
    ):
        super().__init__(size, observation_space, action_space, prediction_depth)
        self.tree = SumTree(size)
        self.alpha = alpha
        self.eps = eps

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        idx = self.buffer.add(
            obs_t,
            nxtobs_t,
            actions=action,
            rewards=reward,
            terminated=terminated,
        )
        self.tree.add(self.tree.max(), idx)
        if terminated or truncated:
            self.buffer.on_episode_end(terminated)

    def sample(self, batch_size: int, beta=0.4):
        idxs = np.zeros((batch_size), dtype=np.int32)
        priorities = np.zeros((batch_size), dtype=np.float32)
        buffer_idxs = np.zeros((batch_size), dtype=np.int32)
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, buffer_idx = self.tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            buffer_idxs[i] = buffer_idx
        obs, data, terminated, filled, _ = self.buffer.sample(
            buffer_idxs, traj_len=self.prediction_depth
        )
        weight = np.power(self.tree.n_entries * priorities / self.tree.total(), -beta)
        weight_max = np.power(self.tree.n_entries * self.tree.min() / self.tree.total(), -beta)
        weights = weight / weight_max
        return {
            "obses": obs,
            **data,
            "terminateds": terminated,
            "filled": filled,
            "weights": weights,
            "indexes": idxs,
        }

    def update_priorities(self, indexes, priorities):
        priorities = np.power(priorities + self.eps, self.alpha)
        for idx, p in zip(indexes, priorities):
            self.tree.update(idx, p)


if __name__ == "__main__":
    buffer = PrioritizedTransitionReplayBuffer(
        20, observation_space=[(4,)], action_space=1, prediction_depth=5, alpha=0.6, eps=1e-6
    )
    for idx in range(10):
        buffer.add(
            [np.arange(idx, idx + 4)],
            idx + 1,
            idx + 1,
            [np.arange(idx + 1, idx + 5)],
            False,
            truncated=False,
        )
    sample = buffer.sample(1)
    print("shape : ")
    for k, v in sample.items():
        if v is not None and isinstance(v, list):
            for idx, a in enumerate(v):
                print(k + str(idx), a.shape)
        else:
            print(k, v.shape)

    print("sample : ", sample)
    buffer.add(
        [np.arange(idx, idx + 4)],
        idx + 1,
        idx + 1,
        [np.arange(idx + 1, idx + 5)],
        True,
        truncated=True,
    )

    buffer.update_priorities(sample["indexes"], np.random.rand(5))
