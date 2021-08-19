from yolo.yolo_models import YOLO

IMAGE_SIZE = 16
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

yolov4 = YOLO(INPUT_SHAPE)
yolov4.show()
