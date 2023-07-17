import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial

LOG_STD_MAX = 2
LOG_STD_MIN = -5
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN)/2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN)/2.0

class Actor(hk.Module):
	def __init__(self,action_size,action_type,node=256,hidden_n=2):
		super(Actor, self).__init__()
		self.action_size = action_size
		self.action_type = action_type
		self.node = node
		self.hidden_n = hidden_n
		self.layer = hk.Linear
		self.layer = partial(self.layer, w_init=hk.initializers.VarianceScaling(scale=2), b_init=hk.initializers.VarianceScaling(scale=2))
		
	def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
			mlp = hk.Sequential(
				[
					self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
				]
				)(feature)
			if self.action_type == 'discrete':
				action_probs = self.layer(self.action_size[0])(mlp)
				return action_probs
			elif self.action_type == 'continuous':
				mu = self.layer(self.action_size[0])(mlp)
				#log_std = LOG_STD_MEAN + LOG_STD_SCALE*jax.nn.tanh(self.layer(self.action_size[0])(mlp) / LOG_STD_SCALE)
				#log_std = jnp.clip(self.layer(self.action_size[0])(mlp), -20, 2)
				log_std = hk.get_parameter("log_std", [1,self.action_size[0]], jnp.float32, init=jnp.zeros)
				return mu, log_std 
		
class Critic(hk.Module):
	def __init__(self,node=256,hidden_n=2):
		super(Critic, self).__init__()
		self.node = node
		self.hidden_n = hidden_n
		self.layer = hk.Linear
		
	def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
		net = hk.Sequential(
			[
				self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
			] + 
			[
				self.layer(1)
			]
			)(feature)
		return net