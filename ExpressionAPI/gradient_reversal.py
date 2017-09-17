import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine.topology import Layer


class ReverseGradient(object):
    num_calls = 0
    '''Flips the sign of incoming gradient during training.'''
    def __init__(self, hp_lambda):
        self.hp_lambda = hp_lambda

    def __call__(self, x):
        grad_name = "GradientReversal%d" % ReverseGradient.num_calls
    
        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * self.hp_lambda]

        g = K.get_session().graph
        
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(x)
        
        ReverseGradient.num_calls += 1
        
        return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self._hp_lambda = hp_lambda
        self.hp_lambda = K.variable(hp_lambda)
        self.supports_masking = False
        self.op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self._trainable_weights = []

    def call(self, x, mask=None):
        return self.op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

