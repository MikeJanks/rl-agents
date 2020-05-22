import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions

from .utils.mish import Mish

class sac(tf.keras.Model):
    ''' Soft Actor Critic '''
    def __init__(self, action_size, observation_shape, is_discrete, gamma=0.99, alpha=0.2, log_std_min=-20, log_std_max=2, **kargs):

        super(sac, self).__init__()

        ''' Hyperparameters '''
        self.observation_shape = observation_shape
        self.is_discrete = is_discrete
        self.action_size = action_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gamma       = gamma
        self.target_ent  = -np.prod(action_size)
        # self.target_ent  = np.log(action_size) if is_discrete else -np.prod(action_size)
        self.log_alpha   = tf.Variable(0, dtype=tf.float64, trainable=True)
        self.alpha       = tf.Variable(0, dtype=tf.float64, trainable=False)
        
        

        ''' Networks '''
        self.actor_net  = self.build_actor(observation_shape, action_size)
        self.critic_net_1 = self.build_critic(observation_shape, action_size)
        self.critic_net_2 = self.build_critic(observation_shape, action_size)
        
        

        ''' Target Networks '''
        self.target_actor_net  = self.build_actor(observation_shape, action_size)
        self.target_critic_net_1 = self.build_critic(observation_shape, action_size)
        self.target_critic_net_2 = self.build_critic(observation_shape, action_size)
        
        self.target_actor_net.set_weights(self.actor_net.get_weights())
        self.target_critic_net_1.set_weights(self.critic_net_1.get_weights())
        self.target_critic_net_2.set_weights(self.critic_net_2.get_weights())
        
        
        
    def build_actor(self, state_size, action_size):
        s  = keras.Input(state_size)
        
        if len(self.observation_shape) == 3:
            s0 = layers.Conv2D(32, 8, 4, activation=Mish(), kernel_initializer='lecun_normal')(s)
            s1 = layers.Conv2D(32, 4, 2, activation=Mish(), kernel_initializer='lecun_normal')(s0)
            s2 = layers.Conv2D(32, 3, 1, activation=Mish(), kernel_initializer='lecun_normal')(s1)
            h0 = layers.Flatten()(s2)
            
        else:
            h0 = layers.Dense(256, activation=Mish(), kernel_initializer='lecun_normal')(s)
            
            
        h1 = layers.Dense(256, activation=Mish(), kernel_initializer='lecun_normal')(h0)
        mu = layers.Dense(action_size, activation='linear', kernel_initializer='lecun_normal')(h1)
        std= layers.Dense(action_size, activation='linear', kernel_initializer='lecun_normal')(h1)
        return tf.keras.Model(inputs=s, outputs=[mu, std])
    
        
    def build_critic(self, state_size, action_size):
        s  = keras.Input(state_size)
        
        if len(self.observation_shape) == 3:
            s0 = layers.Conv2D(32, 8, 4, activation=Mish(), kernel_initializer='lecun_normal')(s)
            s1 = layers.Conv2D(32, 4, 2, activation=Mish(), kernel_initializer='lecun_normal')(s0)
            s2 = layers.Conv2D(32, 3, 1, activation=Mish(), kernel_initializer='lecun_normal')(s1)
            h0 = layers.Flatten()(s2)
            h1 = layers.Dense(256, activation='linear', kernel_initializer='lecun_normal')(h0)
            
        else:
            h0 = layers.Dense(256, activation=Mish(), kernel_initializer='lecun_normal')(s)
            h1 = layers.Dense(256, activation='linear', kernel_initializer='lecun_normal')(h0)
        
        
        a  = keras.Input(action_size)
        a0 = layers.Dense(256, activation='linear', kernel_initializer='lecun_normal')(a)
        
        h2 = layers.Activation(Mish())(layers.Concatenate()([h1, a0]))
        h3 = layers.Dense(256, activation=Mish(), kernel_initializer='lecun_normal')(h2)
        o  = layers.Dense(1, activation='linear', kernel_initializer='lecun_normal')(h3)
        return tf.keras.Model(inputs=[s,a], outputs=o) 


    @tf.function
    def get_action(self, actor_network, state):
        mean, log_std = actor_network(state)
        std           = tf.exp(log_std)
        
        if self.is_discrete:
            logits = mean
            
            uniform_noise   = tf.random.uniform(shape=tf.shape(logits), minval=0., maxval=1., dtype=tf.float64)
            gumbel_noise    = -tf.math.log(-tf.math.log(uniform_noise + 1e-10) + 1e-10)
            noisy_logits    = logits + gumbel_noise
            actions         = tf.nn.softmax(noisy_logits)
            # actions         = tf.nn.softmax(logits)
            
            
            log_prob        = -tf.reduce_sum(-actions * tf.nn.log_softmax(logits), axis=-1) # The Entropy
            
            return actions, log_prob
        
        else:
            normal        = tfd.Normal(mean, std)
            z             = normal.sample()
            actions       = tf.math.tanh(z)
            log_prob      = normal.log_prob(z)
            log_prob      = tf.reduce_sum(log_prob - tf.math.log(1 - actions ** 2 + 1e-10), axis=-1) # The Entropy
            
            return actions, log_prob
        
    
    @tf.function
    def get_value(self, critic_network, state, logits):
        q_value = critic_network([state, logits])
        
        return q_value
    
    
    
    @tf.function
    def actor_loss(self, states):
        logits, log_prob = self.get_action(self.actor_net, states)
        
        q_value_1         = self.get_value(self.critic_net_1, states, logits)
        q_value_2         = self.get_value(self.critic_net_2, states, logits)
        q_value           = tf.minimum(q_value_1, q_value_2)
        
        actor_loss = tf.reduce_mean(self.alpha * log_prob - q_value)
        alpha_loss = tf.reduce_mean(self.log_alpha * tf.stop_gradient(-log_prob - self.target_ent))
        
        return actor_loss, alpha_loss
        
    
    
    @tf.function
    def critic_loss(self, states, actions, logits, rewards, next_states, dones):
        q_value_1      = self.get_value(self.critic_net_1, states, logits)
        q_value_2      = self.get_value(self.critic_net_2, states, logits)
        
        next_logits, next_log_prob = self.get_action(self.target_actor_net, next_states)
        next_q_value_1              = self.get_value(self.target_critic_net_1, next_states, next_logits)
        next_q_value_2              = self.get_value(self.target_critic_net_2, next_states, next_logits)
        next_q_value                = tf.minimum(next_q_value_1, next_q_value_2)
        next_q_value                = next_q_value - self.alpha * next_log_prob
        
        target_q    = tf.expand_dims(rewards, axis=-1) + self.gamma * next_q_value * tf.expand_dims(dones, axis=-1)
        
        loss_1      = tf.keras.losses.mse(y_true=target_q, y_pred=q_value_1)
        loss_2      = tf.keras.losses.mse(y_true=target_q, y_pred=q_value_2)
        loss_1      = tf.reduce_mean(loss_1)
        loss_2      = tf.reduce_mean(loss_2)
        return loss_1, loss_2
        
        
        
            