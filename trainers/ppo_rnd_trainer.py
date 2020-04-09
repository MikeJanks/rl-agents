import tensorflow as tf

import numpy as np
from copy import deepcopy
from time import time
from runstats import Statistics

from agents.ppo_intris_value import ppo_intris_value
from agents.random_network_distilation import random_network_distilation
from agents.utils.gae import get_gaes



class Trainer(tf.keras.Model):
    def __init__(self, actions, epochs=3, batch_size=128, update_num=256,
                    gamma_e=0.99, gamma_i=0.99, lambda_=.95, learning_rate=0.0003,
                    grad_clip=0.5, **kargs):
        super(Trainer, self).__init__()

        self.agent_ppo=ppo_intris_value(actions, **kargs)
        self.agent_rnd=random_network_distilation(**kargs)

        self.epochs=epochs
        self.batch_size=batch_size
        self.update_num=update_num
        self.gamma_e=gamma_e
        self.gamma_i=gamma_i
        self.lambda_=lambda_
        self.grad_clip=grad_clip
        self.init_replay_buffer={
            'states': [],
            'next_states': [],
            'logits': [],
            'actions': [],
            'values_e': [],
            'values_i': [],
            'rewards_e': [],
            'rewards_i': [],
            'rewards_i_raw': [],
            'dones': [],
        }
        self.replay_buffer=deepcopy(self.init_replay_buffer)
        self.running_stats = Statistics()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-10)
        
    def reset_replay_buffer(self):
        self.replay_buffer=deepcopy(self.init_replay_buffer)


    def action(self, states):
        logits, values_e, values_i = self.agent_ppo(np.array([states]))

        values_e = tf.squeeze(values_e).numpy()
        values_i = tf.squeeze(values_i).numpy()
        logits   = tf.squeeze(logits).numpy()
        action   = tf.random.categorical([logits], 1)
        action   = tf.squeeze(action).numpy()

        self.replay_buffer['states'].append(states)
        self.replay_buffer['values_e'].append(values_e)
        self.replay_buffer['values_i'].append(values_i)
        self.replay_buffer['logits'].append(logits)
        self.replay_buffer['actions'].append(action)

        return action


    def add(self, next_state, rewards_e, done):
        rewards_i_raw = self.agent_rnd(np.array([next_state])).numpy()
        
        self.running_stats.push(rewards_i_raw)
        # mean        = self.running_stats.mean()
        # stddev      = self.running_stats.stddev(0)
        rewards_i   = rewards_i_raw
        # rewards_i  *= 10
        # rewards_i   = rewards_i - mean
        # rewards_i   = rewards_i / (stddev + 1e-10)
        
        self.replay_buffer['next_states'].append(next_state)
        self.replay_buffer['rewards_e'].append(-int(done))
        self.replay_buffer['rewards_i'].append(rewards_i)
        self.replay_buffer['rewards_i_raw'].append(rewards_i_raw)
        self.replay_buffer['dones'].append(int(not done))



    def print_stats(self):
        print(f"actions:   {self.replay_buffer['actions'][-1]}")
        print(f"logits:    {self.replay_buffer['logits'][-1]}")
        # print(f"mean: {self.running_stats.mean()}")
        # print(f"stddev: {self.running_stats.stddev(0)}")
        print(f"rewards_i_raw: {self.replay_buffer['rewards_i_raw'][-1]}")
        # print(f"rewards_i: {self.replay_buffer['rewards_i'][-1]/self.running_stats.stddev(0)}")
        print(f"values_i:  {self.replay_buffer['values_i'][-1]}")



    def update(self):
        if len(self.replay_buffer['actions']) % self.update_num==0:
            self.print_stats()
            start_update=time()
            _, next_value_e, next_value_i = self.agent_ppo(np.array([self.replay_buffer['next_states'][-1]]))

            init_returns_e, init_advantages_e = get_gaes(np.array(self.replay_buffer['rewards_e']),
                                                        np.array(self.replay_buffer['values_e']),
                                                        np.array([next_value_e]),
                                                        np.array(self.replay_buffer['dones']),
                                                        gamma=self.gamma_e, lambda_=self.lambda_)

            init_returns_i, init_advantages_i = get_gaes(np.array(self.replay_buffer['rewards_i'])*1000,#)/self.running_stats.stddev(0),
                                                        np.array(self.replay_buffer['values_i']),
                                                        np.array([next_value_i]),
                                                        np.array(self.replay_buffer['dones']),
                                                        gamma=self.gamma_i, lambda_=self.lambda_)

            init_advantages = init_advantages_e + init_advantages_i
            init_advantages = (init_advantages - np.mean(init_advantages)) / (np.std(init_advantages) + 1e-10)

            for _ in range(self.epochs):
                for i in range(0, len(self.replay_buffer['actions']), self.batch_size):
                    states      = np.array(self.replay_buffer['states'][i:i+self.batch_size])
                    next_states = np.array(self.replay_buffer['next_states'][i:i+self.batch_size])
                    actions     = np.array(self.replay_buffer['actions'][i:i+self.batch_size])
                    old_logits  = np.array(self.replay_buffer['logits'][i:i+self.batch_size])
                    old_values_e= np.array(self.replay_buffer['values_e'][i:i+self.batch_size])
                    old_values_i= np.array(self.replay_buffer['values_i'][i:i+self.batch_size])
                    returns_e   = init_returns_e[i:i+self.batch_size]
                    returns_i   = init_returns_i[i:i+self.batch_size]
                    advantages  = init_advantages[i:i+self.batch_size]
                    
                    self.step(states, next_states, actions, old_logits, old_values_e, old_values_i, returns_e, returns_i, advantages)
            self.reset_replay_buffer()
            print(time()-start_update)


    @tf.function
    def step(self, states, next_states, actions, old_logits, old_values_e, old_values_i, returns_e, returns_i, advantages):
        a_one_hot = tf.one_hot(actions, self.agent_ppo.action_size, axis=-1, dtype=tf.float64)
        with tf.GradientTape() as tape:
            logits, values_e, values_i = self.agent_ppo(states)
            values_e = tf.squeeze(values_e, axis=-1)
            values_i = tf.squeeze(values_i, axis=-1)

            value_loss_e = self.agent_ppo.value_loss_fn(returns_e, values_e, old_values_e)
            value_loss_i = self.agent_ppo.value_loss_fn(returns_i, values_i, old_values_i)
            policy_loss  = self.agent_ppo.policy_loss_fn(a_one_hot, logits, old_logits, advantages)
            
            rnd_loss     = self.agent_rnd.rnd_loss_fn(states)

            loss = policy_loss + value_loss_e + value_loss_i + rnd_loss

        grads = tape.gradient(loss, self.trainable_variables)
        # grads, _grad_norm = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))