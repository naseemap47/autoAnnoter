from transformers import pipeline
from anot_utils import save_yolo
from PIL import Image
import cv2
import json
from tqdm import tqdm
import os
import glob
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-p", "--prompt", type=str, required=True,
                help="path to prompt.json")
ap.add_argument("-bt", "--box_thld", type=float, default=0.1,
                help="Box Threshold")
# ap.add_argument("-tt", "--txt_thld", type=str, default=0.25,
#                 help="text threshold")
args = vars(ap.parse_args())


# OWL-ViT
detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

# Read prompt.json
txt_prompt = json.load(open(args['prompt']))
TEXT_PROMPT = ', '.join([str(elem) for elem in txt_prompt])

img_list = sorted(glob.glob(os.path.join(args["dataset"], '*.jpg')) + \
    glob.glob(os.path.join(args["dataset"], '*.jpeg')) + \
    glob.glob(os.path.join(args["dataset"], '*.png')))

for i in tqdm(range(len(img_list))):
    folder_name, file_name = os.path.split(img_list[i])
    img = cv2.imread(img_list[i])
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictions = detector(
        Image.fromarray(img),
        candidate_labels=TEXT_PROMPT,
    )
    phrases = []
    bbox_list = []
    for i in predictions:
        score, label, box = i['score'], i['label'], i['box']
        if score > args["box_thld"]:
            phrases.append(label)
            bbox_list.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])

    class_list = [int(list(txt_prompt).index(value.lstrip())+1) for value in phrases]
    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)

# Save Labe Map
with open(os.path.join(args['dataset'], 'classes.txt'), "w") as output:
    for i in txt_prompt.keys():
        output.write(f'{txt_prompt[i]}\n')
print(f"[INFO] Saved Labelmap to: {os.path.join(args['dataset'], 'classes.txt')}")
