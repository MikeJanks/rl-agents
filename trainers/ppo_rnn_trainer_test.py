import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import numpy as np
from copy import deepcopy
from time import time

from agents.ppo_rnn_test import ppo_rnn
from agents.utils.gae import get_gaes



class Trainer(tf.keras.Model):
    def __init__(self, actions, epochs=3, batch_size=64, update_num=512,
                    gamma=0.99, lambda_=.95, learning_rate=0.0001,
                    grad_clip=0.5, **kargs):
        super(Trainer, self).__init__()

        self.agent=ppo_rnn(actions, **kargs)

        self.epochs=epochs
        self.batch_size=batch_size
        self.update_num=update_num
        self.gamma=gamma
        self.lambda_=lambda_
        self.grad_clip=grad_clip
        self.init_replay_buffer={
            'count': 0,
            'masks': [],
            'states': [],
            'next_states': [],
            'logits': [],
            'actions': [],
            'values': [],
            'rewards': [],
            'dones': [],
        }
        self.replay_buffer=deepcopy(self.init_replay_buffer)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-10)
        

    def reset_replay_buffer(self):
        self.replay_buffer=deepcopy(self.init_replay_buffer)


    def action(self, states, hiddens):
        logits, values, hiddens = self.agent(np.array([[states]]), hiddens)

        values = tf.squeeze(values).numpy()
        logits = tf.squeeze(logits).numpy()
        action = tf.random.categorical([logits], 1)
        action = tf.squeeze(action).numpy()

        return action, values, logits, hiddens



    def add(self, masks, state, action, logits, value, next_state, reward, done):
        self.replay_buffer['count']+=1
        self.replay_buffer['masks'].append(masks)
        self.replay_buffer['states'].append(state)
        self.replay_buffer['actions'].append(action)
        self.replay_buffer['logits'].append(logits)
        self.replay_buffer['values'].append(value)
        self.replay_buffer['next_states'].append(next_state)
        self.replay_buffer['rewards'].append(reward)
        self.replay_buffer['dones'].append(int(not done))



    def update(self, hiddens):
        if self.replay_buffer['count'] == self.update_num:
            print(self.replay_buffer["logits"][0])
            start_update=time()
            _, next_value, _ = self.agent(np.array([[self.replay_buffer['next_states'][-1][-1]]]), hiddens)
            
            init_returns, init_advantages = get_gaes(np.array(self.replay_buffer['rewards']),
                                                np.array(self.replay_buffer['values']),
                                                np.array([next_value]),
                                                np.array(self.replay_buffer['dones']),
                                                gamma=self.gamma, lambda_=self.lambda_)

            init_advantages = (init_advantages - np.mean(init_advantages)) / (np.std(init_advantages) + 1e-10)

            for _ in range(self.epochs):
                for i in range(0, len(self.replay_buffer['actions']), self.batch_size):
                    masks       = np.array(self.replay_buffer['masks'][i:i+self.batch_size])
                    states      = np.array(self.replay_buffer['states'][i:i+self.batch_size])
                    next_states = np.array(self.replay_buffer['next_states'][i:i+self.batch_size])
                    actions     = np.array(self.replay_buffer['actions'][i:i+self.batch_size])
                    old_logits  = np.array(self.replay_buffer['logits'][i:i+self.batch_size])
                    old_values  = np.array(self.replay_buffer['values'][i:i+self.batch_size])
                    returns     = np.array(init_returns[i:i+self.batch_size])
                    advantages  = np.array(init_advantages[i:i+self.batch_size])

                    hiddens     = self.agent.initial_hidden(states.shape[0])

                    self.step(masks, hiddens, states, next_states, actions, old_logits, old_values, returns, advantages)
            self.reset_replay_buffer()
            print(time()-start_update)


    @tf.function
    def step(self, masks, hiddens, states, next_states, actions, old_logits, old_values, returns, advantages):
        a_one_hot = tf.one_hot(actions, self.agent.action_size, axis=-1, dtype=tf.float64)
        with tf.GradientTape() as tape:
            logits, values, hiddens = self.agent(states, hiddens)
            values = tf.squeeze(values, axis=-1)

            value_loss = self.agent.value_loss_fn(returns, values, old_values)
            policy_loss = self.agent.policy_loss_fn(a_one_hot, logits, old_logits, advantages)

            loss = policy_loss + value_loss

        grads = tape.gradient(loss, self.trainable_variables)
        grads, _grad_norm = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return hiddens