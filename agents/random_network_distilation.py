import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from .utils.mish import Mish

class random_network_distilation(tf.keras.Model):
    ''' Proximal Policy Optimization '''
    def __init__(self, embed_dim=128, **kargs):
        super(random_network_distilation, self).__init__()

        self.predictor_fn = keras.Sequential()
        self.predictor_fn.add(layers.Conv2D(32, 8, 4, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='predictor_fn_Conv2D_1'))
        self.predictor_fn.add(layers.Conv2D(64, 4, 2, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='predictor_fn_Conv2D_2'))
        self.predictor_fn.add(layers.Conv2D(64, 3, 1, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='predictor_fn_Conv2D_3'))
        self.predictor_fn.add(layers.Flatten())
        self.predictor_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal', name='predictor_fn_Dense_1'))
        self.predictor_fn.add(layers.Dense(embed_dim, activation='linear'))

        self.target_fn = keras.Sequential()
        self.target_fn.add(layers.Conv2D(32, 8, 4, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='target_fn_Conv2D_1'))
        self.target_fn.add(layers.Conv2D(64, 4, 2, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='target_fn_Conv2D_2'))
        self.target_fn.add(layers.Conv2D(64, 3, 1, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='target_fn_Conv2D_3'))
        self.target_fn.add(layers.Flatten())
        self.target_fn.add(layers.Dense(embed_dim, activation='linear'))
        self.target_norm = tf.keras.layers.LayerNormalization()

        self.target_fn.trainable=False




    @tf.function
    def __call__(self, next_states):
        pred    = self.predictor_fn(next_states)
        target  = self.target_fn(next_states)
        target  = self.target_norm(target)
        i_reward= tf.reduce_mean(tf.keras.losses.mse(target, pred))

        return i_reward


    @tf.function
    def rnd_loss_fn(self, next_states):
        ''' RND Loss '''
        pred    = self.predictor_fn(next_states)
        target  = self.target_fn(next_states)
        target  = self.target_norm(target)
        loss    = tf.reduce_mean(tf.keras.losses.mse(target, pred))

        return loss