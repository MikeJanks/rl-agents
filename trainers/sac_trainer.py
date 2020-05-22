import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from os import mkdir
import numpy as np
from copy import deepcopy
from time import time

from agents.sac import sac
from agents.utils.gae import get_gaes



class Trainer(tf.keras.Model):
    def __init__(self, actions, observations, is_discrete=False, batch_size=64, mem_size=40000,
                    gamma=0.99, actor_lr=0.0003, critic_lr=0.0003, alpha_lr=0.0003, delay=1, tau=0.005, **kargs):
        super(Trainer, self).__init__()
        
        self.agent          = sac(actions, observations, is_discrete, **kargs)

        self.is_discrete    = is_discrete
        self.mem_size       = mem_size
        self.batch_size     = batch_size
        self.gamma          = gamma
        self.tau            = tau
        self.delay          = delay
        self.counter        = 0
        self.replay_buffer  = {
            'states': np.zeros((mem_size,)+observations),
            'actions': np.zeros((mem_size)) if is_discrete else np.zeros((mem_size, actions)),
            'logits': np.zeros((mem_size, actions)),
            'rewards': np.zeros((mem_size,)),
            'next_states': np.zeros((mem_size,)+observations),
            'dones': np.zeros((mem_size,)),
        }
        self.actor_optimizer    = tf.keras.optimizers.Adam(actor_lr, epsilon=1e-10)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(critic_lr, epsilon=1e-10)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(critic_lr, epsilon=1e-10)
        self.alpha_optimizer    = tf.keras.optimizers.Adam(alpha_lr, epsilon=1e-10)


    # @tf.function
    def action(self, states):
        logits, _ = self.agent.get_action(self.agent.actor_net, np.array([states]))
        
        if self.is_discrete:
            # action = tf.random.categorical(logits, 1)
            action = tf.argmax(logits, axis=-1)
            action = tf.squeeze(action).numpy()
        else:
            action = tf.squeeze(logits).numpy()
            
        logits = tf.squeeze(logits).numpy()
        
        return action, logits



    def add(self, state, action, logits, reward, next_state, done):
        i = self.counter % self.mem_size
        self.replay_buffer['states'][i]      = state
        self.replay_buffer['actions'][i]     = action
        self.replay_buffer['logits'][i]      = logits
        self.replay_buffer['rewards'][i]     = reward
        self.replay_buffer['next_states'][i] = next_state
        self.replay_buffer['dones'][i]       = float(not done)
        self.counter+=1
        
        
    def save_model(self, path):
        try:
            self.save_weights(path, save_format='tf')
            print('Saved!')
        except:
            mkdir(path)
            self.save_weights(path, save_format='tf')
    
    
    def load_model(self, path):
        try:
            self.load_weights(path)
        except:
            print("\n\nWeights not Found\n")
            pass
        
        
        
    def sample(self):
        sample_indx = np.random.choice(self.counter if self.counter < self.mem_size else self.mem_size, self.batch_size, replace=False)
        states      = self.replay_buffer['states'][sample_indx]
        actions     = self.replay_buffer['actions'][sample_indx]
        logits      = self.replay_buffer['logits'][sample_indx]
        rewards     = self.replay_buffer['rewards'][sample_indx]
        next_states = self.replay_buffer['next_states'][sample_indx]
        dones       = self.replay_buffer['dones'][sample_indx]

        return states, actions, logits, rewards, next_states, dones



    def update(self):
        if self.counter > self.batch_size:
            
            states, actions, logits, rewards, next_states, dones = self.sample()

            self.critic_step(states, actions, logits, rewards, next_states, dones)
            self.actor_step(states)
            self.update_weights()


    @tf.function
    def critic_step(self, states, actions, logits, rewards, next_states, dones):
        with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
            loss_1, loss_2 = self.agent.critic_loss(states, actions, logits, rewards, next_states, dones)

        grads_1 = tape_1.gradient(loss_1, self.agent.critic_net_1.trainable_variables)
        grads_2 = tape_2.gradient(loss_2, self.agent.critic_net_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(grads_1, self.agent.critic_net_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(grads_2, self.agent.critic_net_2.trainable_variables))



    @tf.function
    def actor_step(self, states):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as alpha_tape:
            actor_loss, alpha_loss = self.agent.actor_loss(states)

        actor_grads = actor_tape.gradient(actor_loss, self.agent.actor_net.trainable_variables)
        alpha_grads = alpha_tape.gradient(alpha_loss, [self.agent.log_alpha])
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.agent.actor_net.trainable_variables))
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.agent.log_alpha]))
        
        
        
    def update_weights(self):
        self.agent.target_critic_net_1.set_weights( list( map( lambda net, target_net: ((self.tau) * net) + ((1-self.tau) * target_net), self.agent.critic_net_1.get_weights(), self.agent.target_critic_net_1.get_weights()) ) )
        self.agent.target_critic_net_2.set_weights( list( map( lambda net, target_net: ((self.tau) * net) + ((1-self.tau) * target_net), self.agent.critic_net_2.get_weights(), self.agent.target_critic_net_2.get_weights()) ) )
        self.agent.target_actor_net.set_weights( list( map( lambda net, terget_net: ((self.tau) * net) + ((1-self.tau) * terget_net), self.agent.actor_net.get_weights(), self.agent.target_actor_net.get_weights()) ) )
        self.agent.alpha.assign(tf.exp(self.agent.log_alpha))
    
    
    