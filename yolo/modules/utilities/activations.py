import tensorflow_addons as tfa
import tensorflow as tf

class Mish(tf.keras.layers.Layer):
  """
    Create a Mish Activation function.
  """
  def call(self, input):
    x = tfa.activations.mish(input)
    return x