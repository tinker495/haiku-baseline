import numpy as np


class Buffer(object):
    def __init__(self, size: int, obs_dict: dict, state_dict: dict, env_dict: dict):
        self.max_size = size
        self._idx = -1
        self.ep_idx = 0
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.env_dict = env_dict
        self.buffer = self.creat_buffer(size, obs_dict, state_dict, env_dict)

    def creat_buffer(self, size: int, obs_dict: dict, state_dict: dict, env_dict: dict):
        buffer = {}
        for name, data in obs_dict.items():
            buffer[name] = np.zeros((size + 1, *data["shape"]), dtype=data["dtype"])
        for name, data in state_dict.items():
            buffer[name] = np.zeros((size + 1, *data["shape"]), dtype=data["dtype"])
        for name, data in env_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        buffer["terminated"] = np.ones((size, 1), dtype=np.bool_)
        buffer["ep_idx"] = np.ones((size + 1, 1), dtype=np.int32) * -1
        return buffer

    def get_all_transitions(self):
        return self.buffer

    def clear(self):
        self._idx = -1
        self.ep_idx = 0
        self.buffer = self.creat_buffer(
            self.max_size, self.obs_dict, self.state_dict, self.env_dict
        )

    def get_stored_size(self):
        return min(self._idx, self.max_size)

    def update_idx(self):
        if self._idx >= self.max_size - 1:
            return True
        self._idx = self._idx + 1
        return False

    def add(self, obs, state, next_obs, next_state, **kwargs):
        if self.update_idx():
            return
        if self.buffer["ep_idx"][self.roll_idx_m1] != self.ep_idx:
            for idx, k in enumerate(self.obs_dict.keys()):
                self.buffer[k][self.roll_idx] = obs[idx]
            for idx, k in enumerate(self.state_dict.keys()):
                self.buffer[k][self.roll_idx] = state[idx]
            self.buffer["ep_idx"][self.roll_idx] = self.ep_idx
        for idx, k in enumerate(self.obs_dict.keys()):
            self.buffer[k][self.next_roll_idx] = next_obs[idx]
        for idx, k in enumerate(self.state_dict.keys()):
            self.buffer[k][self.next_roll_idx] = next_state[idx]
        for k, data in kwargs.items():
            self.buffer[k][self.roll_idx] = data
        self.buffer["ep_idx"][self.next_roll_idx] = self.ep_idx

    def on_episode_end(self, truncated):
        if truncated:
            self.update_idx()
            self.buffer["ep_idx"][self.roll_idx] = -1
        self.ep_idx += 1

    @property
    def roll_idx_m1(self):
        return (self._idx - 1) % self.max_size

    @property
    def next_roll_idx(self):
        return self._idx + 1

    @property
    def roll_idx(self):
        return self._idx


class EpochBuffer(object):
    def __init__(
        self,
        epoch_size: int,
        observation_space: list,
        hidden_space: list,
        worker_size=1,
        action_space=1,
    ):
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
        self.env_dict = {
            "action": {"shape": action_space, "dtype": np.float32},
            "reward": {"shape": (), "dtype": np.float32},
        }
        self.state_dict = dict(
            (
                "state{}".format(idx),
                {"shape": o, "dtype": np.float32},
            )
            for idx, o in enumerate(hidden_space)
        )
        self.worker_size = worker_size
        self.local_buffers = [
            Buffer(
                epoch_size,
                self.obsdict,
                self.state_dict,
                self.env_dict,
            )
            for _ in range(worker_size)
        ]

    def add(
        self, obs_t, action, reward, nxtobs_t, terminated, trucated, state_t=[], nextstate_t=[]
    ):
        for w in range(self.worker_size):
            obs = [o[w] for o in obs_t]
            nxtobs = [o[w] for o in nxtobs_t]
            if state_t:
                state = [s[w] for s in state_t]
                nextstate = [s[w] for s in nextstate_t]
            else:
                state = []
                nextstate = []
            self.local_buffers[w].add(
                obs=obs,
                state=state,
                next_obs=nxtobs,
                next_state=nextstate,
                action=action[w],
                reward=reward[w],
                terminated=terminated[w],
            )
            if terminated[w] or trucated[w]:
                self.local_buffers[w].on_episode_end(trucated[w])

    def get_buffer(self):
        transitions = {
            "obses": [],
            "states": [],
            "actions": [],
            "rewards": [],
            "ep_idxs": [],
            "terminateds": [],
        }
        for w in range(self.worker_size):
            trans = self.local_buffers[w].get_all_transitions()
            transitions["obses"].append([trans[o] for o in self.obsdict.keys()])
            transitions["states"].append([trans[s] for s in self.state_dict.keys()])
            transitions["actions"].append(trans["action"])
            transitions["rewards"].append(trans["reward"])
            transitions["ep_idxs"].append(trans["ep_idx"])
            transitions["terminateds"].append(trans["terminated"])
            self.local_buffers[w].clear()
        return transitions
