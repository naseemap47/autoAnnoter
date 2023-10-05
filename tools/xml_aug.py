from data_aug import RandomHorizontalFlip, RandomScale, RandomTranslate, \
    RandomRotate, RandomShear, Resize, RandomHSV, Sequence
import glob
import os
import cv2
import argparse
import numpy as np
from utils import get_xml, write_xml


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-x", "--xml", type=str, required=True,
                help="path to xml/dir")
ap.add_argument("-s", "--save", type=str, required=True,
                help="path to save aug xml")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_to_xml = args['xml']
path_to_save = args['save']

os.makedirs(path_to_save, exist_ok=True)
xml_list = sorted(glob.glob(f'{path_to_xml}/*.xml'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')
img_list = sorted(img_full_list)

c = 0
for xml_path, img_path in zip(xml_list, img_list):
    c += 1
    path_to_dir, img_name = os.path.split(img_path)
    img = cv2.imread(img_path)

    # Read XML
    bbox_list, class_list = get_xml(xml_path)
    if len(bbox_list) > 0:
        # Augumentation
        bbox_list = np.array(bbox_list, dtype=np.float64)
        if c%3 == 0:
            img_aug, bbox_aug = RandomHorizontalFlip(1)(img.copy(), bbox_list.copy())
        elif c%4 == 0:
            img_aug, bbox_aug = RandomScale(0.3, diff = True)(img.copy(), bbox_list.copy())
        elif c%5 == 0:
            img_aug, bbox_aug = RandomRotate(20)(img.copy(), bbox_list.copy())
        elif c%6 == 0:
            img_aug, bbox_aug = RandomShear(0.2)(img.copy(), bbox_list.copy())
        else:
            bbox_aug = []
    
        if len(bbox_aug) > 0:
            # save image
            cv2.imwrite(f"{path_to_save}/{img_name}", img_aug)
            # Save XML
            xml_name = f"{os.path.splitext(img_name)[0]}.xml"
            path_to_xml_save = f"{path_to_save}/{xml_name}"
            write_xml(img_path, bbox_aug, class_list, path_to_xml_save)
