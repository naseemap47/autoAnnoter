from GroundingDINO.groundingdino.util.inference import load_model, predict
from anot_utils import save_yolo, box_cxcywh_to_xyxy
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
import numpy as np
import glob
import json
import torch
import cv2
import os
import wget
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-p", "--prompt", type=str, required=True,
                help="path to prompt.json")
ap.add_argument("-bt", "--box_thld", type=float, default=0.35,
                help="Box Threshold")
ap.add_argument("-tt", "--txt_thld", type=str, default=0.25,
                help="text threshold")
args = vars(ap.parse_args())


CONFIG_PATH = os.path.join('GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
if not os.path.exists("groundingdino_swint_ogc.pth"):
    print(
        f"[INFO] GroundingDINO Model NOT Found!!! \n \
        [INFO] Downloading GroundingDINO Model..."
    )
    wget.download("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"

transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BOX_TRESHOLD = args['box_thld']
TEXT_TRESHOLD = args['txt_thld']
txt_prompt = json.load(open(args['prompt']))
TEXT_PROMPT = ', '.join([str(elem) for elem in txt_prompt])

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

img_list = glob.glob(os.path.join(args["dataset"], '*.jpg')) + \
    glob.glob(os.path.join(args["dataset"], '*.jpeg')) + \
    glob.glob(os.path.join(args["dataset"], '*.png'))

for img_path in img_list:
    folder_name, file_name = os.path.split(img_path)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    image = np.asarray(img)
    image_transformed, _ = transform(img, None)
    bbox_list, logits, phrases = predict(
        model=model, 
        image=image_transformed, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    class_list = [int(list(txt_prompt).index(value)+1) for value in phrases]
    bbox_list = bbox_list * torch.Tensor([w, h, w, h])
    bbox_list = box_cxcywh_to_xyxy(boxes=bbox_list).numpy()
    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
    print(f'Successfully Annotated {file_name}')

# Save Labe Map
with open(os.path.join(args['dataset'], 'classes.txt'), "w") as output:
    for i in txt_prompt.keys():
        output.write(f'{txt_prompt[i]}\n')
print(f"[INFO] Saved Labelmap to: {os.path.join(args['dataset'], 'classes.txt')}")
