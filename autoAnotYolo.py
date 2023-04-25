import cv2
from utils.hubconf import custom
import argparse
import glob
import os
from anot_utils import save_yolo, get_BBoxYOLOv7, get_BBoxYOLOv8
from ultralytics import YOLO


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-mt", "--model_type", type=str, required=True,
                choices=['yolov7', 'yolov8'],
                help="Choose YOLO model")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to best.pt (YOLOv7) model")
ap.add_argument("-c", "--confidence", type=float, required=True,
                help="Model detection Confidence (0<confidence<1)")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
model_type = args['model_type']
path_or_model = args['model']
detect_conf = args['confidence']

remove_list = ['XUV400', 'notXUV400']

if model_type == 'yolov7':
    # Load YOLOv7 Model
    model = custom(path_or_model=path_or_model)
if model_type == 'yolov8':
    # Load YOLOv8 Model
    model = YOLO(path_or_model)

# Class Names
class_name_list = [x for _, x in model.names.items()]

img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
    glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
    glob.glob(os.path.join(path_to_dir, '*.png'))

for img in img_list:
    folder_name, file_name = os.path.split(img)
    image = cv2.imread(img)
    h, w, c = image.shape

    if model_type == 'yolov7':
        bbox_list, class_list, confidence = get_BBoxYOLOv7(image, model, detect_conf)
    if model_type == 'yolov8':
        bbox_list, class_list, confidence = get_BBoxYOLOv8(image, model, detect_conf, class_name_list, remove_list)

    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
    print(f'Successfully Annotated {file_name}')
print('YOLOv7-Auto_Annotation Successfully Completed')
