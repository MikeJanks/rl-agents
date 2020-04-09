import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from .utils.mish import Mish

class random_network_distilation(tf.keras.Model):
    ''' Proximal Policy Optimization '''
    def __init__(self, embed_dim=128, **kargs):
        super(random_network_distilation, self).__init__()

        self.e_predictor_fn = keras.Sequential()
        self.e_predictor_fn.add(layers.Conv2D(32, 8, 4, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='e_predictor_fn_Conv2D_1'))
        self.e_predictor_fn.add(layers.Conv2D(64, 4, 2, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='e_predictor_fn_Conv2D_2'))
        self.e_predictor_fn.add(layers.Conv2D(64, 3, 1, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='e_predictor_fn_Conv2D_3'))
        self.e_predictor_fn.add(layers.Flatten())
        self.e_predictor_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal', name='e_predictor_fn_Dense_1'))
        # self.e_predictor_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal', name='e_predictor_fn_Dense_2'))
        self.e_predictor_fn.add(layers.Dense(embed_dim, activation='linear', kernel_initializer='lecun_normal'))

        # self.b_predictor_fn = keras.Sequential()
        # self.b_predictor_fn.add(layers.Conv2D(32, 8, 4, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='b_predictor_fn_Conv2D_1'))
        # self.b_predictor_fn.add(layers.Conv2D(64, 4, 2, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='b_predictor_fn_Conv2D_2'))
        # self.b_predictor_fn.add(layers.Conv2D(64, 3, 1, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='b_predictor_fn_Conv2D_3'))
        # self.b_predictor_fn.add(layers.Flatten())
        # self.b_predictor_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal', name='b_predictor_fn_Dense_1'))
        # # self.b_predictor_fn.add(layers.Dense(128, activation=Mish(), kernel_initializer='lecun_normal', name='b_predictor_fn_Dense_2'))
        # self.b_predictor_fn.add(layers.Dense(embed_dim, activation='linear', kernel_initializer='lecun_normal'))

        self.target_fn = keras.Sequential()
        self.target_fn.add(layers.Conv2D(32, 8, 4, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='target_fn_Conv2D_1'))
        self.target_fn.add(layers.Conv2D(64, 4, 2, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='target_fn_Conv2D_2'))
        self.target_fn.add(layers.Conv2D(64, 3, 1, padding='same', activation=Mish(), kernel_initializer='lecun_normal', name='target_fn_Conv2D_3'))
        self.target_fn.add(layers.Flatten())
        self.target_fn.add(layers.Dense(embed_dim, activation='linear', kernel_initializer='lecun_normal'))
        self.target_norm = tf.keras.layers.LayerNormalization(axis=1, center=True, scale=True)

        self.target_fn.trainable=False




    @tf.function
    def __call__(self, next_states):
        e_pred  = self.e_predictor_fn(next_states)
        # b_pred  = self.b_predictor_fn(next_states)
        target  = self.target_fn(next_states)
        excited = tf.reduce_mean(tf.keras.losses.mse(target, e_pred))
        # boredom = tf.reduce_mean(tf.keras.losses.mse(target, b_pred))

        return excited


    @tf.function
    def rnd_loss_fn(self, next_states):
        ''' RND Loss '''
        e_pred  = self.e_predictor_fn(next_states)
        # b_pred  = self.b_predictor_fn(next_states)
        target  = self.target_fn(next_states)
        excited = tf.reduce_mean(tf.keras.losses.mse(target, e_pred))
        # boredom = tf.reduce_mean(tf.keras.losses.mse(target, b_pred))

        return  excited