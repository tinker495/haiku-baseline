import base64
import multiprocessing as mp
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import ray

from jax_baselines.common.cpprb_buffers import ReplayBuffer


@ray.remote(num_cpus=1)
class Ape_X_Worker(object):
    encoded = base64.b64encode(mp.current_process().authkey)

    def __init__(self, env_name_) -> None:
        mp.current_process().authkey = base64.b64decode(self.encoded)
        from jax_baselines.common.atari_wrappers import get_env_type, make_wrap_atari

        self.env_type, self.env_id = get_env_type(env_name_)
        if self.env_type == "atari_env":
            self.env = make_wrap_atari(env_name_, clip_rewards=True)
        else:
            self.env = gym.make(env_name_)

    def get_info(self):
        return {
            "observation_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "env_type": self.env_type,
            "env_id": self.env_id,
        }

    def run(
        self,
        local_size,
        buffer_info,
        model_builder,
        actor_builder,
        param_server,
        logger_server,
        update,
        stop,
        eps=0.05,
    ):
        try:
            gloabal_buffer, env_dict, n_s = buffer_info
            local_buffer = ReplayBuffer(local_size, env_dict=env_dict, n_s=n_s)
            preproc, model = model_builder()
            (
                get_abs_td_error,
                actor,
                get_action,
                random_action,
                key_seq,
            ) = actor_builder()

            get_abs_td_error = jax.jit(partial(get_abs_td_error, model, preproc))
            actor = jax.jit(partial(actor, model, preproc))
            _get_action = partial(get_action, actor)
            get_action = random_action

            state, info = self.env.reset()
            have_original_reward = "original_reward" in info.keys()
            have_lives = "lives" in info.keys()
            if have_original_reward:
                original_score = 0
            score = 0
            state = [np.expand_dims(state, axis=0)]
            params = ray.get(param_server.get_params.remote())
            eplen = 0
            episode = 0
            if eps is None:
                rw_label = "env/episode_reward"
                if have_original_reward:
                    original_rw_label = "env/original_reward"
                len_label = "env/episode_len"
                to_label = "env/time_over"
            else:
                rw_label = f"env/episode_reward/eps{eps:.2f}"
                if have_original_reward:
                    original_rw_label = f"env/original_reward/eps{eps:.2f}"
                len_label = f"env/episode_len/eps{eps:.2f}"
                to_label = f"env/time_over/eps{eps:.2f}"

            while not stop.is_set():
                if update.is_set():
                    params = ray.get(param_server.get_params.remote())
                    update.clear()
                    get_action = _get_action

                eplen += 1
                actions = get_action(params, state, eps, next(key_seq))
                next_state, reward, terminated, truncated, info = self.env.step(actions)
                next_state = [np.expand_dims(next_state, axis=0)]
                local_buffer.add(
                    state, actions, reward, next_state, terminated or truncated, truncated
                )
                if have_original_reward:
                    original_score += info["original_reward"]
                score += reward
                state = next_state

                if terminated or truncated:
                    local_buffer.episode_end()
                    if logger_server is not None:
                        log_dict = {
                            rw_label: score,
                            len_label: eplen,
                            to_label: 1 - terminated,
                        }
                        if have_original_reward:
                            if have_lives:
                                if info["lives"] == 0:
                                    log_dict[original_rw_label] = original_score
                                    original_score = 0
                            else:
                                log_dict[original_rw_label] = original_score
                                original_score = 0
                        logger_server.log_worker.remote(log_dict, episode)
                    score = 0
                    eplen = 0
                    episode += 1
                    state, info = self.env.reset()
                    state = [np.expand_dims(state, axis=0)]

                if len(local_buffer) >= local_size:
                    transition = local_buffer.get_buffer()
                    local_buffer.clear()
                    abs_td_error = get_abs_td_error(
                        params,
                        **local_buffer.conv_transitions(transition),
                        key=next(key_seq),
                    )
                    gloabal_buffer.add(**transition, priorities=abs_td_error)
        finally:
            if stop.is_set():
                print("worker stoped")
            else:
                stop.set()
        return None
