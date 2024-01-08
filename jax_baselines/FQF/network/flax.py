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

        self.pi_mtx = jax.lax.stop_gradient(
            jnp.expand_dims(jnp.pi * (jnp.arange(0, 128, dtype=np.float32) + 1), axis=(0, 2))
        )  # [ 1 x 128 x 1]

    @nn.compact
    def __call__(self, feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape  # [ batch x feature]

        tau = jnp.expand_dims(tau, axis=1)  # [ batch x 1 x tau]
        costau = jnp.cos(tau * self.pi_mtx)  # [ batch x 128 x tau]

        def qnet(feature, costau):  # [ batch x feature], [ batch x 128 ]
            quantile_embedding = nn.Sequential([self.layer(feature_shape[1]), jax.nn.relu])(
                costau
            )  # [ batch x feature ]
            mul_embedding = feature * quantile_embedding  # [ batch x feature ]
            if self.hidden_n != 0:
                mul_embedding = nn.Sequential(
                    [
                        self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                        for i in range(2 * self.hidden_n)
                    ]
                )(mul_embedding)
            if not self.dueling:
                q_net = self.layer(
                    self.action_size[0],
                    kernel_init=clip_uniform_initializers(-0.03, 0.03),
                )(mul_embedding)
                return q_net
            else:
                v = self.layer(1, kernel_init=clip_uniform_initializers(-0.03, 0.03))(mul_embedding)
                a = self.layer(
                    self.action_size[0],
                    kernel_init=clip_uniform_initializers(-0.03, 0.03),
                )(mul_embedding)
                q = v + a - jnp.max(a, axis=1, keepdims=True)
                return q

        out = jax.vmap(qnet, in_axes=(None, 2), out_axes=2)(
            feature, costau
        )  # [ batch x action x tau ]
        return out


class FractionProposal(nn.Module):
    support_size: int
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        batch = feature.shape[0]
        log_probs = jax.nn.log_softmax(
            nn.Sequential(
                [nn.Dense(self.node) if i % 2 == 0 else nn.relu for i in range(2 * self.hidden_n)]
                + [
                    nn.Dense(
                        self.support_size, kernel_init=jax.nn.initializers.variance_scaling(0.01)
                    )
                ],
            )(feature),
            axis=-1,
        )
        probs = jnp.exp(log_probs)
        tau_0 = jnp.zeros((batch, 1), dtype=np.float32)
        tau_1_N = jnp.cumsum(probs, axis=1)
        tau = jnp.concatenate((tau_0, tau_1_N), axis=1)
        tau_hat = jax.lax.stop_gradient((tau[:, :-1] + tau[:, 1:]) / 2.0)
        entropy = -jnp.sum(log_probs * probs, axis=-1, keepdims=True)
        return tau, tau_hat, entropy


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, n_support, policy_kwargs
):
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

        def __call__(self, x, tau):
            x = self.preproc(x)
            return self.qnet(x, tau)

        def preprocess(self, x):
            return self.preproc(x)

        def q(self, x, tau):
            return self.qnet(x, tau)

    def model_builder(key=None, print_model=False):
        model = Merged()
        fqf = FractionProposal(n_support)
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        fqf_fn = get_apply_fn_flax_module(fqf)
        model_fn = get_apply_fn_flax_module(model, model.q)
        if key is not None:
            tau = jax.random.uniform(key, (1, 2))
            params = model.init(
                key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space], tau
            )
            out = preproc_fn(
                params, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            fqf_param = fqf.init(key, out)
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", params)
                print_param("", fqf_param)
                print("------------------------------------------------------")
            return preproc_fn, model_fn, fqf_fn, params, fqf_param
        else:
            return preproc_fn, model_fn, fqf_fn

    return model_builder
