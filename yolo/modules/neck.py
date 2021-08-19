from .utilities.nn_blocks import SPP_block, convbn
import tensorflow as tf


def neck_yolov4(input, name, shapes = False):
    """
        Create the Neck YOLOv4 structure.

        Uses a SPP additional block, for fixed outpuut 
        dimensions, and better activation map generalization. 

        PANet maintains fine grained detailes from prior layers
        and solve the vanishing gradient problem for long convolutional
        chains.

        Paramaters
        ----------
        input : tf.keras.layers.Input
        
        name : str
        
        [shapes : bool]


        Returns 
        ---------

    """

    input_1 = tf.keras.layers.Input(input[0].shape[1:]) #input[0]
    input_2 = tf.keras.layers.Input(input[1].shape[1:])
    input_3 = tf.keras.layers.Input(input[2].shape[1:])


    spp = SPP_block(input_3) # Maxpoolings [x, Maxpool1, Maxpool2, Maxpool3]

    # 
    # PANet 
    #

    x = convbn(spp, filters=512, kernel_size=(1,1))
    x = convbn(x, filters=1024, kernel_size=(3,3))
    
    output_3 = convbn(x, filters=512, kernel_size=(1,1)) # Output 
    
    x = convbn(output_3, filters=256, kernel_size=(1,1))
    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = convbn(input_2, filters=256, kernel_size=(1,1))

    x = tf.keras.layers.Concatenate()([x, upsampled])

    # Fire module
    x = convbn(x, filters=256, kernel_size=(1,1))
    x = convbn(x, filters=512, kernel_size=(3,3))
    x = convbn(x, filters=256, kernel_size=(1,1))
    x = convbn(x, filters=512, kernel_size=(3,3))
    
    output_2 = convbn(x, filters=256, kernel_size=(1,1)) # Output 
    
    x = convbn(output_2, filters=128, kernel_size=(1,1))

    upsampled = tf.keras.layers.UpSampling2D()(x)


    x = convbn(input_1, filters=128, kernel_size=(1,1))
    x = tf.keras.layers.Concatenate()([x, upsampled])

    # Fire module
    x = convbn(x, filters=128, kernel_size=(1,1))
    x = convbn(x, filters=256, kernel_size=(3,3))
    x = convbn(x, filters=128, kernel_size=(1,1))
    x = convbn(x, filters=256, kernel_size=(3,3))

    output_1 = convbn(x, filters=128, kernel_size=(1,1)) # Output 

    return tf.keras.Model(
        [input_1, input_2, input_3], [output_1, output_2, output_3], name=name
    )