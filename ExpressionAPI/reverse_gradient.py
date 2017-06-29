import tensorflow as tf
from tensorflow.python.framework import ops


class GradientReversal(object):

    def __init__(self):
        self.n_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.n_calls
        self.n_calls += 1

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        return y
    
reverse_gradient = GradientReversal()
