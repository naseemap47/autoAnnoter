from data_aug import RandomHorizontalFlip, RandomScale, RandomTranslate, \
    RandomRotate, RandomShear, Resize, RandomHSV, Sequence, Rotate, \
    Translate, Scale, Shear
from numpy.random import choice
import glob
import os
import cv2
import argparse
import numpy as np
from tool_utils import get_xml, write_xml
import yaml


def generate_anot_RandomHSV(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, hue=None, saturation=None, brightness=None):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = RandomHSV(hue, saturation, brightness)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_Rotate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, angle:int):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = Rotate(angle)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_RandomRotate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, angle:int):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = RandomRotate(angle)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_Translate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, translate_x:float, translate_y:float):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = Translate(translate_x, translate_y)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_RandomTranslate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, translate:float):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = RandomTranslate(translate)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_Scale(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, scale_x:float, scale_y:float):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = Scale(scale_x, scale_y)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_RandomScale(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, scale:float):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = RandomScale(scale)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_Shear(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, shear:float):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = Shear(shear)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anot_RandomShear(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, shear:float):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = RandomShear(shear)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


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


with open('default.yaml', 'r') as f:
	data = yaml.load(f, Loader=yaml.SafeLoader)
print(data)

os.makedirs(path_to_save, exist_ok=True)
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')
img_list = sorted(img_full_list)

# image HSV-Hue augmentation
total_prob = int((data['hsv_h']['prob'])*len(img_list))
hsv_h_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_h_img, path_to_xml, hue=data['hsv_h']['hue'], path_to_save=path_to_save, name_id='hsv_h')

# image HSV-Saturation augmentation
total_prob = int((data['hsv_s']['prob'])*len(img_list))
hsv_s_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_s_img, path_to_xml, saturation=data['hsv_s']['saturation'], path_to_save=path_to_save, name_id='hsv_s')

# image HSV-Value (brightness) augmentation
total_prob = int((data['hsv_v']['prob'])*len(img_list))
hsv_v_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_v_img, path_to_xml, brightness=data['hsv_v']['brightness'], path_to_save=path_to_save, name_id='hsv_v')

# Mixed image HSV augmentation (Mixed HSV-Hue, HSV-Saturation and HSV-Value (brightness))
total_prob = int((data['hsv']['prob'])*len(img_list))
hsv_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_img, path_to_xml, hue=data['hsv']['hue'], saturation=data['hsv']['saturation'], brightness=data['hsv']['brightness'], path_to_save=path_to_save, name_id='hsv')

# image rotation augmentation
total_prob = int((data['degrees']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Rotate(rot_img, path_to_xml, path_to_save, 'rot', data['degrees']['deg'])

# image Random rotation augmentation
total_prob = int((data['degrees_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomRotate(rot_img, path_to_xml, path_to_save, 'rot_random', data['degrees_random']['deg'])

# image Translate augmentation
total_prob = int((data['translate']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Translate(rot_img, path_to_xml, path_to_save, 'trans', data['translate']['translate_x'], data['translate']['translate_y'])

# image Random Translate augmentation
total_prob = int((data['translate_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomTranslate(rot_img, path_to_xml, path_to_save, 'trans_random', data['translate_random']['translate'])

# image Scale augmentation
total_prob = int((data['scale']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Scale(rot_img, path_to_xml, path_to_save, 'scale', data['scale']['scale_x'], data['scale']['scale_y'])

# image Random Scale augmentation
total_prob = int((data['scale_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomScale(rot_img, path_to_xml, path_to_save, 'scale_random', data['scale_random']['scale'])

# image Shear augmentation
total_prob = int((data['shear']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Shear(rot_img, path_to_xml, path_to_save, 'shear', data['shear']['shear'])

# image Random Shear augmentation
total_prob = int((data['shear_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomShear(rot_img, path_to_xml, path_to_save, 'shear_random', data['shear_random']['shear'])

