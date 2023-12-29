import haiku as hk
import jax
import numpy as np
import ray

from jax_baselines.common.base_classes import select_optimizer
from jax_baselines.common.utils import key_gen
from jax_baselines.IMPALA.cpprb_buffers import ImpalaBuffer


@ray.remote(num_gpus=1)
class Impala_Trainer(object):
    def __init__(
        self,
        learning_rate,
        minibatch_size,
        replay=False,
        batch_size=1024,
        model_builder=None,
        train_builder=None,
        optimizer="rmsprop",
        logger_server=None,
        key=42,
    ):
        self.buffer = ImpalaBuffer(replay)
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.key_seq = key_gen(key)
        self.preproc, self.model = model_builder()
        self._preprocess, self._train_step = train_builder()
        self.optimizer = select_optimizer(optimizer, learning_rate, 1e-2 / batch_size)
        self.logger_server = logger_server

    def setup_model(self):
        pre_param = self.preproc.init(
            next(self.key_seq),
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        model_param = self.model.init(
            next(self.key_seq),
            self.preproc.apply(
                pre_param,
                None,
                [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
            ),
        )
        self.params = hk.data_structures.merge(pre_param, model_param)

        self.opt_state = self.optimizer.init(self.params)

        self._preprocess = jax.jit(self._preprocess)
        self._train_step = jax.jit(self._train_step)

    def get_params(self):
        return self.params

    def append_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self):
        if self.replay is not None:
            data = self.buffer.sample(self.batch_size)
        else:
            data = self.buffer
            data = {k: np.array([data[k]]) for k in data.keys()}
        self.params, loss = self._train_step(self.params, self.opt_state, next(self.key_seq))
        self.logger_server.append_loss.remote(loss)
