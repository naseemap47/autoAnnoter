from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert
from anot_utils import save_yolo
import supervision as sv
import cv2
import numpy as np
import GroundingDINO.groundingdino.datasets.transforms as T
from typing import Tuple
from PIL import Image
import glob
import os
import json
import torch
from torch import Tensor
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
args = vars(ap.parse_args())


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes



CONFIG_PATH = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
txt_prompt = json.load(open('prompt.json'))
TEXT_PROMPT = ', '.join([str(elem) for elem in txt_prompt])
print(TEXT_PROMPT)
# TEXT_PROMPT = "glass most to the right"

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
    print(bbox_list, logits, phrases)
    class_list = [int(list(txt_prompt).index(value)+1) for value in phrases]
    print(class_list)
    bbox_list = bbox_list * torch.Tensor([w, h, w, h])
    bbox_list = box_cxcywh_to_xyxy(boxes=bbox_list).numpy()
    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
    print(f'Successfully Annotated {file_name}')

# Save Labe Map
with open(f"{args['dataset']}/classes.txt", "w") as output:
    for i in txt_prompt.keys():
        output.write(f'{i}\n')
print(f'[INFO] Saved Labelmap to: {args["dataset"]}/classes.txt')
