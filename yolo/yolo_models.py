from .modules.backbone import csp_darknet53
from .modules.neck import neck_yolov4
from .modules.head import yolo_head

import tensorflow as tf 
import numpy as np

class YOLO:
    """
        YOLOv4 Implementation Model. 

        Takes de input shape for the constructor, and 
        creates an instance of the YOLOV4 net. 
        
        @version: 0.1v
    """
    def __init__(self, input_shape):
        anchors = np.random.rand(3, 2)

        # Define the Model phases
        self.backbone = csp_darknet53(input_shape, name = "CSPDarknet53")
        self.neck = neck_yolov4(self.backbone.outputs, name = "SPP/PAN")
        self.head = yolo_head(self.neck.outputs, anchors = anchors, num_classes = 8) # , name = "Detector") #, anchors = anchors, num_classes = 8, name = "Detector")

        # Declare Inputs 
        self.inputs = tf.keras.layers.Input(input_shape, name = "Inputs")

          # Process each phase
        low = self.backbone(self.inputs)
        print("low -> mid", end=' ->')
        mid = self.neck(low)
        print(" high")
        high = self.head(mid)

        self.Model = tf.keras.Model(inputs = self.inputs, outputs = high)

    def predict(self, input):
        return self.Model.predict(input)
    

    def show(self):
        self.Model.summary()