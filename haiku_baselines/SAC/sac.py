import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from haiku_baselines.SAC.network import Actor, Critic, Value
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import soft_update, convert_jax

class SAC(Deteministic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.99, learning_rate=3e-4, buffer_size=100000, train_freq=1, gradient_steps=1, ent_coef = 'auto', 
                 batch_size=32, policy_delay = 3, n_step = 1, learning_starts=1000, target_network_update_tau=5e-4,
                 prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
                 prioritized_replay_eps=1e-6, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
        
        super(SAC, self).__init__(env, gamma, learning_rate, buffer_size, train_freq, gradient_steps, batch_size,
                 n_step, learning_starts, target_network_update_tau, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps,
                 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed, optimizer)
        
        self.policy_delay = policy_delay
        self.ent_coef = ent_coef
        self.target_entropy = -np.sqrt(np.prod(self.action_size).astype(np.float32))
        
        if _init_setup_model:
            self.setup_model() 
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
        self.preproc = hk.transform(lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x))
        self.actor = hk.transform(lambda x: Actor(self.action_size,**self.policy_kwargs)(x))
        self.critic = hk.transform(lambda x,a: Critic(**self.policy_kwargs)(x,a))
        self.value = hk.transform(lambda x: Value(**self.policy_kwargs)(x))
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature, self.actor.apply(actor_param, None, feature))
        value_param = self.value.init(next(self.key_seq), feature)
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param, value_param)
        self.target_params = self.params
        
        self.opt_state = self.optimizer.init(self.params)
        
        print("----------------------model----------------------")
        print(jax.tree_map(lambda x: x.shape, pre_param))
        print(jax.tree_map(lambda x: x.shape, actor_param))
        print(jax.tree_map(lambda x: x.shape, critic_param))
        print("-------------------------------------------------")

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)
        self._actor_train_step = jax.jit(self._actor_train_step)
        
    def _get_update_data(self,params,feature,key = None) -> jnp.ndarray:
        mu, log_std = self.actor.apply(params, key, feature)
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key,std.shape)
        pi = jax.nn.tanh(x_t)
        var = (std ** 2)
        log_prob = jnp.sum(-jnp.square(x_t - mu) / (2 * var) - jnp.log(std) - jnp.log(jnp.sqrt(2 * jnp.pi)) - jnp.log(1 - jnp.square(jnp.pi) + 1e-6),keepdims=True)
        return pi, log_prob, mu, log_std, std
        
    def _get_actions(self, params, obses, key = None) -> jnp.ndarray:
        mu, log_std = self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses)))
        std = jnp.exp(log_std)
        pi = jax.nn.tanh(mu + std * jax.random.normal(key,std.shape))
        return pi
    
    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque), np.mean(self.lossque)
                                    )
    
    def actions(self,obs,steps):
        actions = self._get_actions(self.params,obs, next(self.key_seq))
        return actions
    
    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)
            
            self.params, self.target_params, self.opt_state, loss, new_priorities = \
                self._train_step(self.params, self.target_params, self.opt_state, next(self.key_seq),
                                 **data)
                
            t_mean = None
            if steps % self.policy_delay == 0:
                self.params, self.opt_state, t_mean = \
                self._actor_train_step(self.params, self.opt_state,data['obses'],next(self.key_seq))
            
            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data['indexes'], new_priorities)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            if t_mean is not None:
                self.summary.add_scalar("loss/targets", t_mean, steps)
            
        return loss

    def _train_step(self, params, target_params, opt_state, key, 
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); not_dones = 1.0 - dones
        targets = self._target(target_params, rewards, nxtobses, not_dones, key)
        (critic_loss,abs_error), critic_grad = jax.value_and_grad(self._critic_loss,has_aux = True)(params, obses, actions, targets, weights, key)
        updates, opt_state = self.optimizer.update(critic_grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = soft_update(params, target_params, self.target_network_update_tau)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error + self.prioritized_replay_eps
        return params, target_params, opt_state, critic_loss, new_priorities
    
    def _actor_train_step(self,params,opt_state,obses,key):
        actor_loss, actor_grad = jax.value_and_grad(self._actor_loss)(params, obses, key)
        updates, opt_state = self.optimizer.update(actor_grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, actor_loss
    
    def _critic_loss(self, params, obses, actions, targets, weights, key):
        feature = self.preproc.apply(params, key, obses)
        q1, q2 = self.critic.apply(params, key, feature, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        return jnp.mean(weights*jnp.square(error1)) + jnp.mean(weights*jnp.square(error2)), jnp.abs(error1)
    
    def _actor_loss(self, params, obses, key, ent_coef):
        feature = self.preproc.apply(params, key, convert_jax(obses))
        policy, log_prob, mu, log_std, std = self._get_update_data(params, obses, key)
        vals, _ = self.critic.apply(jax.lax.stop_gradient(params), key, feature, policy)
        return jnp.mean(ent_coef * log_prob - vals)
    
    def _target(self, target_params, rewards, nxtobses, not_dones, key):
        next_feature = self.preproc.apply(target_params, key, nxtobses)
        v = self.value.apply(target_params, key, next_feature)
        return (not_dones * v * self._gamma) + rewards
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="SAC",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)