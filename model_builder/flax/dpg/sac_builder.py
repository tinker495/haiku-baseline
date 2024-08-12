import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_uniform_initializers
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN) / 2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN) / 2.0


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        linear = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(
                    self.action_size[0] * 2, kernel_init=clip_uniform_initializers(-0.03, 0.03)
                ),
            ]
        )(feature)
        mu, log_std = jnp.split(linear, 2, axis=-1)
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(
            log_std / LOG_STD_SCALE
        )  # jnp.clip(log_std,LOG_STD_MIN,LOG_STD_MAX)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        q_net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_uniform_initializers(-0.03, 0.03))]
        )(concat)
        return q_net


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x):
                feature = self.preprocess(x)
                mu, log_std = self.actor(feature)
                return mu, log_std
            
            def preprocess(self, x):
                x = self.preproc(x)
                return x
            
            def actor(self, x):
                return self.act(x)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.crit1 = Critic(**policy_kwargs)
                self.crit2 = Critic(**policy_kwargs)

            def __call__(self, x, a):
                return (self.crit1(x, a), self.crit2(x, a))

        model_actor = Merged_Actor()
        preproc_fn = get_apply_fn_flax_module(model_actor, model_actor.preprocess)
        actor_fn = get_apply_fn_flax_module(model_actor, model_actor.actor)
        model_critic = Merged_Critic()
        critic_fn = get_apply_fn_flax_module(model_critic)
        if key is not None:
            policy_params = model_actor.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            critic_params = model_critic.init(
                key,
                preproc_fn(policy_params, key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]),
                np.zeros((1, *action_size), dtype=np.float32),
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", policy_params)
                print_param("", critic_params)
                print("------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        else:
            return preproc_fn, actor_fn, critic_fn

    return model_builder
