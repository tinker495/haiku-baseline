import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.common.utils import print_param
from jax_baselines.model.flax.apply import get_apply_fn_flax_module
from jax_baselines.model.flax.initializers import clip_uniform_initializers
from jax_baselines.model.flax.layers import NoisyDense
from jax_baselines.model.flax.Module import PreProcess


class Model(nn.Module):
    action_size: int
    node: int
    hidden_n: int
    noisy: bool
    dueling: bool

    def setup(self) -> None:
        if not self.noisy:
            self.layer = nn.Dense
        else:
            self.layer = NoisyDense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if self.hidden_n != 0:
            feature = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
            )(feature)
        if not self.dueling:
            q_net = self.layer(
                self.action_size[0], kernel_init=clip_uniform_initializers(-0.03, 0.03)
            )(feature)
            return q_net
        else:
            v = self.layer(1, kernel_init=clip_uniform_initializers(-0.03, 0.03))(feature)
            a = self.layer(self.action_size[0], kernel_init=clip_uniform_initializers(-0.03, 0.03))(
                feature
            )
            return v + a - jnp.max(a, axis=1, keepdims=True)


def model_builder_maker(observation_space, action_space, dueling_model, param_noise, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    class Merged(nn.Module):
        def setup(self):
            self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
            self.qnet = Model(
                action_space, dueling=dueling_model, noisy=param_noise, **policy_kwargs
            )

        def __call__(self, x):
            x = self.preproc(x)
            return self.qnet(x)

        def preprocess(self, x):
            return self.preproc(x)

        def q(self, x):
            return self.qnet(x)

    def model_builder(key=None, print_model=False):
        model = Merged()
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        model_fn = get_apply_fn_flax_module(model, model.q)
        if key is not None:
            params = model.init(
                key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", params)
                print("------------------------------------------------------")
            return preproc_fn, model_fn, params
        else:
            return preproc_fn, model_fn

    return model_builder
