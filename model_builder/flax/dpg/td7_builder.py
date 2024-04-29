import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_uniform_initializers
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param


def avgl1norm(x, epsilon=1e-6):
    return x / (jnp.abs(x).mean(axis=-1, keepdims=True) + epsilon)


class Encoder(nn.Module):
    node: int = 256
    hidden_n: int = 3
    layer: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoder = nn.Sequential(
            [
                self.layer(self.node) if i % 2 == 0 else jax.nn.elu
                for i in range(2 * self.hidden_n - 1)
            ]
        )(feature)
        return avgl1norm(encoder)


class Action_Encoder(nn.Module):
    node: int = 256
    hidden_n: int = 3
    layer: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([zs, action], axis=1)
        zsa = nn.Sequential(
            [
                self.layer(self.node) if i % 2 == 0 else jax.nn.elu
                for i in range(2 * self.hidden_n - 1)
            ]
        )(concat)
        return zsa


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        a0 = avgl1norm(self.layer(self.node)(feature))
        embed_concat = jnp.concatenate([a0, zs], axis=1)
        action = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(self.action_size[0], kernel_init=clip_uniform_initializers(-0.03, 0.03)),
                jax.nn.tanh,
            ]
        )(embed_concat)
        return action


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = nn.Dense

    @nn.compact
    def __call__(
        self, feature: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        embedding = jnp.concatenate([zs, zsa], axis=1)
        q0 = avgl1norm(self.layer(self.node)(concat))
        embed_concat = jnp.concatenate([q0, embedding], axis=1)
        q_net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.elu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_uniform_initializers(-0.03, 0.03))]
        )(embed_concat)
        return q_net


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class Merge_encoder(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.enc = Encoder()
                self.act_enc = Action_Encoder()

            def __call__(self, x, a):
                feature = self.preprocess(x)
                zs = self.encoder(feature)
                zsa = self.action_encoder(zs, a)
                return feature, zs, zsa

            def preprocess(self, x):
                return self.preproc(x)

            def encoder(self, feature):
                return self.enc(feature)

            def action_encoder(self, zs, a):
                return self.act_enc(zs, a)

        class Merged_actor_critic(nn.Module):
            def setup(self):
                self.act = Actor(action_size, **policy_kwargs)
                self.crit1 = Critic(**policy_kwargs)
                self.crit2 = Critic(**policy_kwargs)

            def __call__(self, feature, zs, zsa):
                action = self.actor(feature, zs)
                q = self.critic(feature, zs, zsa, action)
                return q

            def actor(self, feature, zs):
                return self.act(feature, zs)

            def critic(self, feature, zs, zsa, a):
                return (self.crit1(feature, zs, zsa, a), self.crit2(feature, zs, zsa, a))

        encoder_model = Merge_encoder()
        preproc_fn = get_apply_fn_flax_module(encoder_model, encoder_model.preprocess)
        encoder_fn = get_apply_fn_flax_module(encoder_model, encoder_model.encoder)
        action_encoder_fn = get_apply_fn_flax_module(encoder_model, encoder_model.action_encoder)
        model = Merged_actor_critic()
        actor_fn = get_apply_fn_flax_module(model, model.actor)
        critic_fn = get_apply_fn_flax_module(model, model.critic)
        if key is not None:
            encoder_params = encoder_model.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                np.zeros((1, *action_size), dtype=np.float32),
            )
            params = model.init(
                key,
                *encoder_model.apply(
                    encoder_params,
                    [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                    np.zeros((1, *action_size), dtype=np.float32),
                )
            )

            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", encoder_params)
                print("------------------------------------------------------")
                print_param("", params)
                print("------------------------------------------------------")
            return (
                preproc_fn,
                encoder_fn,
                action_encoder_fn,
                actor_fn,
                critic_fn,
                encoder_params,
                params,
            )
        else:
            return preproc_fn, encoder_fn, action_encoder_fn, actor_fn, critic_fn

    return model_builder
