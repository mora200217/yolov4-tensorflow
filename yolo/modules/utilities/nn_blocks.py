from tensorflow.keras import layers
from .activations import Mish

import tensorflow as tf

## ----------------------------------------------
## -----               HEAD                  ----
## ----------------------------------------------


def residual_block(inputs, n, filters):
  """
    Create a residuual block, that add a prior 
    featuure map with a new convolve one from the 
    previous one. 

    Parameters 
    ----------
    inputs : Tensor 
    n : int 
        Amount of conv -> conv -> Residual blocks 
    filters : (int, int)


  """
  x = inputs
  for iteration in range(n):
      shortcut = x
      
      x = convbn(x, filters, (1, 1))
      x = convbn(x, filters * 2, (3, 3))

      x = layers.Add()([shortcut, x])
  return x


def convbn(inputs, filters, kernel_size = (1, 1), stride = (1, 1)):
  """
    Convolution Batch Normalization.

    Applies the sequence: Conv2D -> Batch Norm -> MISH 
    for a given input

    Parameters 
    ----------
    filters : int
        Amount of filters (channels) for the convolution to apply

    kernel_size : tuple (int, int)
        Size of the sliding window (matrix) for the convolution.
        Always an odd number.

    Stride : tuple (int, int)
        Displacement of the sliding window.

    Returns 
    --------
    x : tf.Tensor 
        Resulting Tensor after the applied sequence.

    Examples
    --------
    >> x = convbn(img, filters = 32, kernel_size = (3, 3), stride = (1, 1))
  """
  x = tf.keras.layers.Conv2D(filters, kernel_size= kernel_size, strides = stride, padding = "same")(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = Mish()(x)

  return x


## ----------------------------------------------
## -----               NECK                  ----
## ----------------------------------------------

def SPP_block(input): 
  """
  Create an Spatial Pyramid Pooling (SPP) Block 
  
  Takes the desired input and passes it through a 
  max pooling stack, to a Dense Layer. 

  Parameters
  ----------
  inputs : tuple (W, H, C)
      Activation Maps 
      
  Returns
  -------
  Keras Model

  """ 
  # Convolution layers 
  x = convbn(input, filters = 1024, kernel_size = (3,3))
  # 11x11x512
  x = convbn(x, filters = 512, kernel_size = (3,3))


  # Max Pool Layers  ----
  # Create vectors abstraction from the inputs feature 
  # Maps and create a final fixed size vector 

  maxpool1 = tf.keras.layers.MaxPool2D((5,5), strides=(1,1), padding = "same")(x)
  maxpool2 = tf.keras.layers.MaxPool2D((9,9), strides=(1,1), padding = "same")(x)
  maxpool3 = tf.keras.layers.MaxPool2D((13,13), strides=(1,1), padding = "same")(x)

  spp = tf.keras.layers.Concatenate()([x, maxpool1, maxpool2, maxpool3])

  return spp


def PAN(input):
  """
  Create an Spatial Pyramid Pooling (SPP) Block 

  Takes the desired input and passes it through a 
  max pooling stack, to a Dense Layer. 

  Parameters
  ----------
  input_shape : tuple (W, H, C)
      Dimensions for the expected input tensor 
      
  Returns
  -------
  Keras Model
  """
  print("Input shape: {}".format(input.shape))

  return None 


## ---------------
## ----------------------------------------------
## -----               HEAD                  ----
## ----------------------------------------------
#@title
def conv_class(input, num_anchors, num_classes):

  # Create n filters based in 1 PC (Probability Class) 
  # value, 4 bounding box parameters (Width, Height, centroid x, centroid y)

  num_filters = num_anchors * (5 + num_classes)
  x = tf.keras.layers.Conv2D(num_filters , kernel_size= 1, strides= 1, padding="same" )(input)

  print("Numanchors: {}".format(num_anchors))
  # 5D Tensor. Intersting 
  x = tf.keras.layers.Reshape((x.shape[1], x.shape[2], num_anchors, num_classes + 5))(x)

  return x 
  

def yolov3_boxes_regression(feats_per_stage, anchors_per_stage):
    # Get Width and height of the feature maps 
    grid_size_x, grid_size_y = feats_per_stage.shape[1], feats_per_stage.shape[2]

    num_classes = feats_per_stage.shape[-1] - 5  # feats.shape[-1] = 4 + 1 + num_classes

    box_xy, box_wh, objectness, class_probs = tf.split(
        feats_per_stage, (2, 2, 1, num_classes), axis=-1
    )

    # Normalize
    box_xy = tf.sigmoid(box_xy) # 0 - 1
    objectness = tf.sigmoid(objectness) # 0 - 1
    
    class_probs = tf.sigmoid(class_probs) # 0 - 1
   

    grid = tf.meshgrid(tf.range(grid_size_y), tf.range(grid_size_x))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gy, gx, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.constant(
        [grid_size_y, grid_size_x], dtype=tf.float32
    )

    box_wh = tf.exp(box_wh) * anchors_per_stage

    # Non maxima suppression bbox characterization 
    box_x1y1 = box_xy - box_wh / 2 # Upper left 
    box_x2y2 = box_xy + box_wh / 2 # Lower Right 

    # Bounding box for nms 
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs


def yolo_nms(
    yolo_feats, 
    yolo_max_boxes, 
    yolo_iou_threshold, 
    yolo_score_threshold
    ):
  
    """ 
    YOLO Non maximum Supression 

    Parameters 
    ----------
    yolo_feats 
      each feat_stage is a tuple with (Bbox, Objectness, Classes)

    conv_class -> nms 

    Tensor W,H, Anchors, Num_classes + 5
    """
    bbox_per_stage, objectness_per_stage, class_probs_per_stage = [], [], []

    # GO Through each stage 
    # stage = scales ? 
    for stage_feats in yolo_feats:
        """
        Stage_feats
              [0] Bbox 
              [1] Objectness 
              [2] Clasess 
        """

        # Num Boxes 
        num_boxes = (
            stage_feats[0].shape[1] * stage_feats[0].shape[2] * stage_feats[0].shape[3]
        ) 

        

        bbox_per_stage.append(
            tf.reshape(
                stage_feats[0],
                (tf.shape(stage_feats[0])[0], num_boxes, stage_feats[0].shape[-1]),
            )
        ) 
        objectness_per_stage.append(
            tf.reshape(
                stage_feats[1],
                (tf.shape(stage_feats[1])[0], num_boxes, stage_feats[1].shape[-1]),
            )
        ) 
        class_probs_per_stage.append(
            tf.reshape(
                stage_feats[2],
                (tf.shape(stage_feats[2])[0], num_boxes, stage_feats[2].shape[-1]),
            )
        )  

    bbox = tf.concat(bbox_per_stage, axis=1)
    objectness = tf.concat(objectness_per_stage, axis=1)
    class_probs = tf.concat(class_probs_per_stage, axis=1)


    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes= tf.expand_dims(bbox, axis=2),
        scores= objectness * class_probs,
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return [boxes, scores, classes, valid_detections]