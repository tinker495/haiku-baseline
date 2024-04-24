import cpprb
import numpy as np


class TransitionRoller(object):
    def __init__(self, obsdict, action_space, prediction_depth=5):
        self.obsdict = obsdict
        self.prediction_depth = prediction_depth
        self.rolldict = dict(
            (k, np.zeros((v["shape"]), dtype=v["dtype"])) for k, v in obsdict.items()
        )
        self.rolldict["action"] = np.zeros((self.prediction_depth, *action_space), dtype=np.float32)
        self.rolldict["reward"] = np.zeros((self.prediction_depth,), dtype=np.float32)
        self.rolldict["terminated"] = np.zeros((self.prediction_depth,), dtype=np.bool_)
        self.rolldict["filled"] = np.zeros((self.prediction_depth,), dtype=np.bool_)
        self._idx = -1

    def __call__(self, obses, action, reward, nxtobses, terminated):
        if self._idx == -1:
            self._idx = 0
            for idx, k in enumerate(self.obsdict.keys()):
                self.rolldict[k][self.obs_idx] = obses[idx]
        for idx, k in enumerate(self.obsdict.keys()):
            self.rolldict[k][self.next_obs_idx] = nxtobses[idx]
        self.rolldict["action"][self.idx] = action
        self.rolldict["reward"][self.idx] = reward
        self.rolldict["terminated"][self.idx] = terminated
        self.rolldict["filled"][self.idx] = not terminated
        self.update_idx()

    def fill_empty(self):
        self.rolldict["filled"][self.idx] = False
        self.update_idx()

    def update_idx(self):
        self._idx = self._idx + 1

    @property
    def obs_roll_idx(self):
        return max(self._idx - self.prediction_depth, 0) % (self.prediction_depth + 1)

    @property
    def next_obs_idx(self):
        return (self._idx + 1) % (self.prediction_depth + 1)

    @property
    def obs_idx(self):
        return self._idx % (self.prediction_depth + 1)

    @property
    def roll_idx(self):
        return max(self._idx - self.prediction_depth, 0) % self.prediction_depth

    @property
    def idx(self):
        return self._idx % self.prediction_depth

    def get_transition(self):
        return dict(
            (k, np.roll(v, -self.obs_roll_idx, axis=0))
            if k in self.obsdict
            else (k, np.roll(v, -self.roll_idx, axis=0))
            for k, v in self.rolldict.items()
        )

    def clear(self):
        self.rolldict = dict((k, np.zeros_like(v)) for k, v in self.rolldict.items())
        self._idx = -1


class TransitionReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        prediction_depth=5,
    ):
        self.max_size = size
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {
                    "shape": (prediction_depth + 1, *o),
                    "dtype": np.uint8 if len(o) >= 3 else np.float32,
                },
            )
            for idx, o in enumerate(observation_space)
        )

        if isinstance(action_space, int):
            action_space = (action_space,)
        self.buffer = cpprb.ReplayBuffer(
            size,
            env_dict={
                **self.obsdict,
                "action": {"shape": (prediction_depth, *action_space), "dtype": np.float32},
                "reward": {"shape": (prediction_depth,), "dtype": np.float32},
                "terminated": {"shape": (prediction_depth,), "dtype": np.bool_},
                "filled": {"shape": (prediction_depth,), "dtype": np.bool_},
            },
        )
        self.roller = TransitionRoller(
            self.obsdict, action_space, prediction_depth=prediction_depth
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
        self.roller(obs_t, action, reward, nxtobs_t, terminated)
        if self.roller._idx >= self.roller.prediction_depth:
            self.buffer.add(**self.roller.get_transition())
        if terminated or truncated:
            for _ in range(self.roller.prediction_depth - 1):
                self.roller.fill_empty()
                self.buffer.add(**self.roller.get_transition())
            self.roller.clear()

    def sample(self, batch_size: int):
        smpl = self.buffer.sample(batch_size)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "terminateds": smpl["terminated"],
            "filled": smpl["filled"],
        }


class PrioritizedTransitionReplayBuffer(TransitionReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        prediction_depth=5,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ):
        self.max_size = size
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {
                    "shape": (prediction_depth + 1, *o),
                    "dtype": np.uint8 if len(o) >= 3 else np.float32,
                },
            )
            for idx, o in enumerate(observation_space)
        )

        if isinstance(action_space, int):
            action_space = (action_space,)
        self.buffer = cpprb.PrioritizedReplayBuffer(
            size,
            env_dict={
                **self.obsdict,
                "action": {"shape": (prediction_depth, *action_space), "dtype": np.float32},
                "reward": {"shape": (prediction_depth,), "dtype": np.float32},
                "terminated": {"shape": (prediction_depth,), "dtype": np.bool_},
                "filled": {"shape": (prediction_depth,), "dtype": np.bool_},
            },
            alpha=alpha,
            eps=eps,
        )
        self.roller = TransitionRoller(
            self.obsdict, action_space, prediction_depth=prediction_depth
        )

    def sample(self, batch_size: int, beta=0.5):
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "terminateds": smpl["terminated"],
            "filled": smpl["filled"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)


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
    buffer.add(
        [np.arange(idx, idx + 4)],
        idx + 1,
        idx + 1,
        [np.arange(idx + 1, idx + 5)],
        True,
        truncated=True,
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

    buffer.update_priorities(sample["indexes"], np.random.rand(5))
