import cv2
import argparse
import glob
import os
from anot_utils import save_yolo, get_BBoxYOLOv8
from ultralytics import YOLO


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to best.pt (YOLOv8) model")
ap.add_argument("-c", "--confidence", type=float, required=True,
                help="Model detection Confidence (0<confidence<1)")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
path_or_model = args['model']
detect_conf = args['confidence']

# Load YOLOv8 Model (best.pt)
model = YOLO(path_or_model)

img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
    glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
    glob.glob(os.path.join(path_to_dir, '*.png'))

for img in img_list:
    folder_name, file_name = os.path.split(img)
    image = cv2.imread(img)
    h, w, c = image.shape
    bbox_list, class_list, confidence = get_BBoxYOLOv8(image, model, detect_conf)
    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)

    print(f'Successfully Annotated {file_name}')

print('YOLOv7-Auto_Annotation Successfully Completed')

