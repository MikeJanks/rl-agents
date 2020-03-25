import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from tensorflow import keras


class Mish(keras.layers.Layer):
    '''
    Mish Activation Function.
    .. math::
        Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * tf.tanh(keras.activations.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape