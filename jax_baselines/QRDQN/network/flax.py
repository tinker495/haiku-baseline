import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.common.utils import print_param
from jax_baselines.model.flax.apply import get_apply_fn_flax_module
from jax_baselines.model.flax.layers import NoisyDense
from jax_baselines.model.flax.Module import PreProcess


class Model(nn.Module):
    action_size: int
    node: int
    hidden_n: int
    noisy: bool
    dueling: bool
    support_n: int

    def setup(self) -> None:
        if not self.noisy:
            self.layer = nn.Dense
        else:
            self.layer = NoisyDense
        if self.dueling:
            # self.value_support_n is divisor of self.support_n and close to sqrt(self.support_n)
            # self.value_support_n * self.action_support_n = self.support_n
            for i in range(int(np.sqrt(self.support_n)) + 1, 0, -1):
                if self.support_n % i == 0:
                    self.value_support_n = i
                    break
            self.action_support_n = self.support_n // self.value_support_n

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q_net = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(self.action_size[0] * self.support_n),
                    lambda x: jnp.reshape(x, (x.shape[0], self.action_size[0], self.support_n)),
                ]
            )(feature)
            return q_net
        else:
            v = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(self.value_support_n),
                    lambda x: jnp.reshape(x, (x.shape[0], 1, 1, self.value_support_n)),
                ]
            )(feature)
            a = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(self.action_size[0] * self.action_support_n),
                    lambda x: jnp.reshape(
                        x, (x.shape[0], self.action_size[0], self.action_support_n, 1)
                    ),
                ]
            )(feature)
            q = v + a - jnp.mean(a, axis=(1, 2, 3), keepdims=True)
            q = jnp.reshape(q, (q.shape[0], self.action_size[0], self.support_n))
            q = jnp.sort(q, axis=2)
            return q


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, support_n, policy_kwargs
):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
        model = Model(
            action_space,
            dueling=dueling_model,
            noisy=param_noise,
            support_n=support_n,
            **policy_kwargs
        )
        preproc_fn = get_apply_fn_flax_module(preproc)
        model_fn = get_apply_fn_flax_module(model)
        if key is not None:
            key1, key2, key3 = jax.random.split(key, 3)
            pre_param = preproc.init(
                key1, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            model_param = model.init(
                key2,
                preproc_fn(
                    pre_param,
                    key3,
                    [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                ),
            )
            params = {"params": {**pre_param["params"], **model_param["params"]}}
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("preprocess", pre_param)
                print_param("model", model_param)
                print("------------------------------------------------------")
            return preproc_fn, model_fn, params
        else:
            return preproc_fn, model_fn

    return model_builder