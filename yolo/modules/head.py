from .utilities.nn_blocks import convbn, conv_class, yolov3_boxes_regression, yolo_nms
import tensorflow as tf
import numpy as np
# Define Anchor Boxes 

ANCHORS = [
           np.array([(120, 240)]), # Small 
           np.array([(120, 240)]),  # Medum 
           np.array([(120, 240)])  # Large
           ]

# -------------------------------
NUM_CLASSES = 8 # Amount of parts
IOU_THRESHOLD = 0.9
YOLO_MAX_BOXES = 3
YOLO_SCORE_THRESHOLD = 0.1
# -------------------------------


def yolo_head(inputs, num_classes, anchors ,yolo_max_boxes=3,
          yolo_iou_threshold=0.4,
          yolo_score_threshold=0.2,
          training = True):
  """

  """

  # Inputs --------------------------------------
  input_1 = tf.keras.Input(shape=inputs[0].shape[1:])
  input_2 = tf.keras.Input(shape=inputs[1].shape[1:])
  input_3 = tf.keras.Input(shape=inputs[2].shape[1:])

  x = convbn(input_1, 256, (3,3))

  # Small scale 
  output_1 = conv_class(
      x, len(ANCHORS[0]), num_classes
  )

  

  x = convbn(
        input_1,
        filters=256,
        kernel_size=(3,3),
        stride=2
  )
   

  x = tf.keras.layers.Concatenate()([x, input_2])

  x = convbn(x, filters=256, kernel_size=(1,1), stride=1)
  x = convbn(x, filters=512, kernel_size=(3,3), stride=1)
  x = convbn(x, filters=256, kernel_size=(1,1), stride=1)
  x = convbn(x, filters=512, kernel_size=(3, 3), stride=1)


  connection = convbn( x, filters=256, kernel_size=(1, 1), stride=1)
  
  x = convbn(connection, filters=512, kernel_size=3, stride=1)

  output_2 = conv_class(x, num_anchors= len(ANCHORS[1]), num_classes=num_classes)

  x = convbn(
      connection,
      filters=512,
      kernel_size=(3,3),
      stride=2)
  
  x = tf.keras.layers.Concatenate()([x, input_3])

  # Convolution Block (Upsample) 512 -> 1024
  x = convbn(x, filters=512, kernel_size=(1,1), stride=1)
  x = convbn(x, filters=1024, kernel_size=(3,3), stride=1)
  x = convbn(x, filters=512, kernel_size=(1,1), stride=1)
  x = convbn(x, filters=1024, kernel_size=(3,3), stride=1)
  x = convbn(x, filters=512, kernel_size=(1,1), stride=1)
  x = convbn(x, filters=1024, kernel_size=(3,3), stride=1)

  output_3 = conv_class(
      x, num_anchors=len(ANCHORS[2]), num_classes=num_classes
  )

 
  if training:
    return tf.keras.Model(
        [input_1, input_2, input_3],
        [output_1, output_2, output_3], # large, medium, small    WxHxAx(1 + 4 + classes)
        # vector -> [1 x 52^2 * 13] 
        name="YOLOv3_head",
    )

  # Ouput_1, Oupuut_2, output_3 -> Tensor 4D (W, H , Anchors, Classes + 5)

  # Predictions for each scale 
  predictions_1 = tf.keras.layers.Lambda( lambda x_input: yolov3_boxes_regression(x_input, anchors[0]), name="yolov3_boxes_regression_small_scale",)(output_1)
  predictions_2 = tf.keras.layers.Lambda( lambda x_input: yolov3_boxes_regression(x_input, anchors[1]), name="yolov3_boxes_regression_medium_scale", )(output_2)
  predictions_3 = tf.keras.layers.Lambda( lambda x_input: yolov3_boxes_regression(x_input, anchors[2]), name="yolov3_boxes_regression_large_scale", )(output_3)
  

  """
  names = ["Bounding Box", "Objecteness", "Classes"]
  for idx, prediction in enumerate(predictions_1): 
    print(names[idx], " ", prediction.shape)
  """
  # Define the final tensor 
  output = tf.keras.layers.Lambda(
      lambda x_input: yolo_nms(
          x_input,
          yolo_max_boxes=YOLO_MAX_BOXES,
          yolo_iou_threshold= IOU_THRESHOLD,
          yolo_score_threshold= YOLO_SCORE_THRESHOLD,
      ),
      name="yolov4_nms",
  )([predictions_1, predictions_2, predictions_3])

  print("Executing nms algorithm")
  # Predictions 
  """
  for idx, prediction in enumerate(output):
    print(f"prediction {idx + 1}: {prediction} /", end = "")
    print(prediction.shape)
    """
  # Output 
  # 1 Boxes 
  # 2 Objectness  
  # 3 Class 

  boxes = output[0] # R3 (None, Anchor Boxes, 4)
  objectness = tf.expand_dims(output[1], -1) #R2 (None, Anchor Boxes)
  classes = tf.expand_dims(output[2], -1) #R2 (None, Classes)

  "(W_g, H_g, anchor box * (4 + 1 + NumClasses) )  "
  # one hot encoding
  
  print("boxes[0]")
  print(boxes[0])

  return tf.keras.Model([input_1, input_2, input_3], output, name="YOLOv3_head")