import cv2
from utils.hubconf import custom
import argparse
import glob
import os
from anot_utils import save_yolo, get_BBoxYOLOv7, get_BBoxYOLOv8, get_BBoxYOLONAS
from ultralytics import YOLO
from super_gradients.training import models
import torch
import yaml


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-mt", "--model_type", type=str, required=True,
                choices=['yolov7', 'yolov8', 'yolonas'],
                help="Choose YOLO model")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to best.pt (YOLOv7) model")

# If its YOLO-NAS its need 2 more args
ap.add_argument("-t", "--type", type=str,
                default='yolo_nas_s',
                help="YOLO-NAS Model type")
ap.add_argument("-y", "--yaml", type=str,
                help="path to data.yaml")


ap.add_argument("-c", "--confidence", type=float, required=True,
                help="Model detection Confidence (0<confidence<1)")
ap.add_argument("-r", "--remove", nargs='+', default=[],
                help="List of classes need to remove")
args = vars(ap.parse_args())


if args['model_type'] == 'yolov7':
    # Load YOLOv7 Model
    model = custom(path_or_model=args['model'])
    # Class Names
    class_name_list = [x for _, x in model.names.items()]
    
if args['model_type'] == 'yolov8':
    # Load YOLOv8 Model
    model = YOLO(args['model'])
    # Class Names
    class_name_list = [x for _, x in model.names.items()]

if args['model_type'] == 'yolonas':
    # Load YOLO-NAS Model
    yaml_params = yaml.safe_load(open(args['yaml'], 'r'))
    model = models.get(
        args['type'], 
        checkpoint_path=args['model'], 
        num_classes=len(yaml_params['names'])
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    # Class Names
    # class_name_list = yaml_params['names']

img_list = glob.glob(os.path.join(args["dataset"], '*.jpg')) + \
    glob.glob(os.path.join(args["dataset"], '*.jpeg')) + \
    glob.glob(os.path.join(args["dataset"], '*.png'))

for img in img_list:
    folder_name, file_name = os.path.split(img)
    image = cv2.imread(img)
    h, w, c = image.shape

    if args['model_type'] == 'yolov7':
        bbox_list, class_list, confidence = get_BBoxYOLOv7(image, model, args['confidence'], class_name_list, args['remove'])
    if args['model_type'] == 'yolov8':
        bbox_list, class_list, confidence = get_BBoxYOLOv8(image, model, args['confidence'], class_name_list, args['remove'])
    if args['model_type'] == 'yolonas':
        bbox_list, class_list, confidence = get_BBoxYOLONAS(image, model, args['confidence'], args['remove'])

    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
    print(f'Successfully Annotated {file_name}')

# Save Labe Map
with open(f"{args['dataset']}/classes.txt", "w") as output:
    for i in class_name_list:
        output.write(f'{i}\n')
print(f'[INFO] Saved Labelmap to: {args["dataset"]}/classes.txt')
print(f"[INFO] {args['model_type']}-Auto_Annotation Successfully Completed")
