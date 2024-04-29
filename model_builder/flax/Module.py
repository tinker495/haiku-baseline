from typing import Callable, List, Tuple

import flax
import flax.linen as nn
import jax.numpy as jnp


def flatten(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape((x.shape[0], -1))


def visual_embedding(mode: str = "normal") -> Callable[[jnp.ndarray], jnp.ndarray]:
    if mode == "normal":
        net = nn.Sequential(
            [
                nn.Conv(
                    32,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    64,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    64,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                flatten,
            ]
        )

    elif mode == "simple":

        net = nn.Sequential(
            [
                nn.Conv(
                    16,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    32,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                flatten,
            ]
        )

    elif mode == "minimum":
        net = nn.Sequential(
            [
                nn.Conv(
                    16,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    32,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                flatten,
            ]
        )
    elif mode == "none":
        net = flatten
    return net


class PreProcess(nn.Module):
    states_size: List[Tuple[int, ...]]
    embedding_mode: str = "normal"

    def setup(self):
        self.embedding = [
            visual_embedding(self.embedding_mode) if len(st) == 3 else lambda x: x
            for st in self.states_size
        ]

    @nn.compact
    def __call__(self, states: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre, x in zip(self.embedding, states)], axis=1)

    @property
    def output_size(self):
        return sum(
            [
                pre(jnp.zeros((1,) + st)).shape[1]
                for pre, st in zip(self.embedding, self.states_size)
            ]
        )
