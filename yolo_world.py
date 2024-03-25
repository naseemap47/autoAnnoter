from anot_utils import save_yolo, save_xml
from ultralytics import YOLO
import argparse
import cv2
import tqdm
import glob
import json
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to data/dir")
ap.add_argument("-p", "--prompt", type=str, required=True,
                help="path to prompt.json")
ap.add_argument("-c", "--conf", type=float, default=0.1,
                help="detection confidence")
ap.add_argument("-m", "--model", type=str,
                choices=[
                    'yolov8s-world.pt', 'yolov8s-worldv2.pt',
                    'yolov8m-world.pt', 'yolov8m-worldv2.pt',
                    'yolov8l-world.pt', 'yolov8l-worldv2.pt',
                    'yolov8x-world.pt', 'yolov8x-worldv2.pt'
                ],
                required= True,
                help="choose model type")
ap.add_argument("-f", "--format", type=float,
                choices=['txt', 'xml'],
                help="annotation format")
args = vars(ap.parse_args())

# Read prompt.json
txt_prompt = json.load(open(args['prompt']))
TEXT_PROMPT = ', '.join([str(elem) for elem in txt_prompt])

img_list = sorted(glob.glob(os.path.join(args["data"], '*.jpg')) + \
    glob.glob(os.path.join(args["data"], '*.jpeg')) + \
    glob.glob(os.path.join(args["data"], '*.png')))

# Initialize a YOLO-World model
model = YOLO(args[['model']])  # or choose yolov8m/l-world.pt
class_names = TEXT_PROMPT.split(',')
print('Class Names: ', class_names)

# Define custom classes
model.set_classes(class_names)

# Execute prediction for specified categories on images
for i in tqdm(range(len(img_list))):
    folder_name, file_name = os.path.split(img_list[i])
    img = cv2.imread(img_list[i])
    h, w, c = img.shape
    # Prediction
    class_list = []
    bbox_list = []
    results = model.predict(img, conf=args['conf'])
    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls
        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            if args['format'] == 'txt':
                class_list.append(int(cs)+1)
            elif args['format'] == 'xml':
                class_list.append(int(cs))
            bbox_list.append([xmin, ymin, xmax, ymax])
    
    # YOLO (TXT) Annotation
    if args['format'] == 'txt':
        save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
    # PASCAL VOC (XML) Annotation
    elif args['format'] == 'xml':
        save_xml(
            folder_name, file_name, img_list[i], w, h, c, 
            bbox_list, class_list, class_names
        )

if args['format'] == 'txt':
    # Save Labe Map
    with open(os.path.join(args['data'], 'classes.txt'), "w") as output:
        for i in class_names:
            output.write(f'{i}\n')
    print(f"[INFO] Saved Labelmap to: {os.path.join(args['data'], 'classes.txt')}")
print('[INFO] YOLO-World autoAnnotation completed ...')
