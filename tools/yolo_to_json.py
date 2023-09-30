import glob
import os
import json
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-t", "--txt", type=str, required=True,
                help="path to txt/dir")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_to_txt = args['txt']
path_to_class = args['classes']

txt_list = sorted(glob.glob(f'{path_to_txt}/*.txt'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

img_list = sorted(img_full_list)
class_names = open(path_to_class, 'r+').read().splitlines()
class_json = []
for i in range(len(class_names)):
    class_ = {
        "id": i,
        "name": class_names[i]
        }
    class_json.append(class_)

img_json_list = []
txt_json_list = []
c_img = 0
c_anot = 0
for txt, img in zip(txt_list, img_list):
    # Image
    folder_name, file_name = os.path.split(img)
    img_file = cv2.imread(img)
    height_n, width_n, depth_n = img_file.shape
    img_dict = {
            "id": c_img,
            "license": 1,
            "file_name": file_name,
            "height": height_n,
            "width": width_n,
        }
    img_json_list.append(img_dict)

    # Annotation
    lines = open(txt, 'r+').read().splitlines()
    obj_list = []
    class_list = []
    for line in lines:
        class_index, x_center, y_center, width, height = line.split()
        xmax = int((float(x_center)*width_n) + (float(width) * width_n)/2.0)
        xmin = int((float(x_center)*width_n) - (float(width) * width_n)/2.0)
        ymax = int((float(y_center)*height_n) + (float(height) * height_n)/2.0)
        ymin = int((float(y_center)*height_n) - (float(height) * height_n)/2.0)
        width_anot = float((xmax - xmin))
        height_anot = float((ymax - ymin))
        # Annotations Dict
        annotation = {
            "id": c_anot,
            "image_id": c_img,
            "category_id": int(class_index),
            "bbox": [
                xmin,
                ymin,
                width_anot,
                height_anot
            ],
            "area": int(width_anot*height_anot),
            "segmentation": [],
            "iscrowd": 0
        }
        txt_json_list.append(annotation)
        c_anot += 1
    c_img += 1

full_json = {
    "categories": class_json,
    "images": img_json_list,
    "annotations": txt_json_list
}
json_object = json.dumps(full_json, indent=3)
with open(f"{path_to_img}/{path_to_img.split('/')[-2]}.json", "w") as outfile:
    outfile.write(json_object)
print(f"[INFO] Saved JSON: {path_to_img}/{path_to_img.split('/')[-2]}.json")
