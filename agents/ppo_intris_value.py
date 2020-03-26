import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from .utils.mish import Mish

class ppo_intris_value(tf.keras.Model):
    ''' Proximal Policy Optimization '''
    def __init__(self, action_size, img_dim=32,
                    policy_clip=0.2, value_clip=0.2,
                    entropy_beta=0.001, val_discount=1.0, **kargs):

        super(ppo_intris_value, self).__init__()

        ''' Hyperparameters '''
        self.action_size  = action_size
        self.policy_clip  = policy_clip
        self.value_clip   = value_clip
        self.entropy_beta = entropy_beta
        self.val_discount = val_discount


        ''' Networks '''
        self.embed_fn = keras.Sequential()
        self.embed_fn.add(layers.Conv2D(32, 8, 4, padding='same', activation=Mish(), kernel_initializer='lecun_normal'))
        self.embed_fn.add(layers.Conv2D(64, 4, 2, padding='same', activation=Mish(), kernel_initializer='lecun_normal'))
        self.embed_fn.add(layers.Conv2D(64, 3, 1, padding='same', activation=Mish(), kernel_initializer='lecun_normal'))
        self.embed_fn.add(layers.Flatten())
        self.embed_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal'))
        self.embed_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal'))

        self.policy_fn = keras.Sequential()
        self.policy_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal'))
        self.policy_fn.add(layers.Dense(self.action_size, activation='linear', kernel_initializer='lecun_normal'))

        self.value_fn = keras.Sequential()
        self.value_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal'))

        self.value_extrins_fn = keras.Sequential()
        self.value_extrins_fn.add(layers.Dense(1, activation='linear', kernel_initializer='lecun_normal'))

        self.value_intrins_fn = keras.Sequential()
        self.value_intrins_fn.add(layers.Dense(1, activation='linear', kernel_initializer='lecun_normal'))




    @tf.function
    def __call__(self, states):
        features = self.embed_fn(states)
        logits   = self.policy_fn(features)
        values   = self.value_fn(features)
        values_e = self.value_extrins_fn(values)
        values_i = self.value_intrins_fn(values)

        return logits, values_e, values_i


    @tf.function
    def value_loss_fn(self, returns, values, old_values):
        ''' Value Loss '''
        values_clip = old_values + tf.clip_by_value(values - old_values, -self.value_clip, self.value_clip)
        val_loss1   = tf.square(values - returns)
        val_loss2   = tf.square(values_clip - returns)
        val_loss    = tf.maximum(val_loss1, val_loss2)
        val_loss    = (self.val_discount/2) * tf.reduce_mean(val_loss)

        return val_loss



    @tf.function
    def policy_loss_fn(self, actions, logits, old_logits, advantages):
        ''' Policy Loss '''
        m_lprobs    = tf.nn.log_softmax(logits)
        m_a_lprobs  = tf.reduce_sum(m_lprobs * actions, axis=-1)

        o_lprobs    = tf.nn.log_softmax(old_logits)
        o_a_lprobs  = tf.reduce_sum(o_lprobs * actions, axis=-1)

        trust       = tf.exp(m_a_lprobs - o_a_lprobs)
        clipped     = tf.clip_by_value(trust, 1-self.policy_clip, 1+self.policy_clip)
        pg_loss     = tf.minimum(trust*advantages, clipped*advantages)
        pg_loss     = -tf.reduce_mean(pg_loss)

        ent_loss    = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1)
        ent_loss    = self.entropy_beta * -tf.reduce_mean(ent_loss)

        return pg_loss + ent_loss