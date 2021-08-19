
import tensorflow as tf
from .utilities.nn_blocks import convbn, residual_block

def csp_darknet53(input_shape, name):
  """
    Backbone: CSP Darknet53

    Returns 
    -------
        Output : array
            [0] output1 - 52x52 
            [2] output2 - 26x26
            [3] output3 - 13x13
  """
  
  
  inputs = tf.keras.layers.Input(shape = input_shape )## 416x416x3
  
  x = convbn(inputs, 32, (1,1)) ## 414x414x32
  
  x = convbn(x, 64, (3,3), stride= (2, 2)) ## 20x410x64
  

  # Blocks 
  
  x = residual_block(x, 1, filters = 32) # 208x208
  
  x = convbn(x, 128, (3, 3), stride = (2, 2)) # 104x104

  x = residual_block(x, 2, filters = 64)
  x = convbn(x, 256, (3, 3), stride = (2, 2)) # 52 x 52

  # ---------------------------------------------------

  output1 = residual_block(x, 8, filters = 128) # Output 1
  x = convbn(output1, 512, (1, 1), stride = (2, 2))

  output2 = residual_block(x, 8, filters = 256) # Output 2
  x = convbn(output2, 1024, (1, 1), stride = (2, 2))

  output3 = residual_block(x, 4, filters = 512) # Output 3
  
  return tf.keras.Model(inputs = inputs, outputs = [output1, output2, output3], name = name)
