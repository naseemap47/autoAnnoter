from data_aug import RandomHorizontalFlip, HorizontalFlip, RandomScale, Scale, RandomTranslate, Translate,\
                    RandomRotate, Rotate, RandomShear, Shear, Resize, RandomHSV, Sequence
import os
import cv2
import numpy as np
from tools.tool_utils import get_xml, write_xml


def generate_anotX_RandomHSV(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, hue=None, saturation=None, brightness=None):
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


def generate_anotX_Rotate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, angle:int):
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


def generate_anotX_RandomRotate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, angle:int):
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


def generate_anotX_Translate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, translate_x:float, translate_y:float):
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


def generate_anotX_RandomTranslate(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, translate:float):
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


def generate_anotX_Scale(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, scale_x:float, scale_y:float):
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


def generate_anotX_RandomScale(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, scale:float):
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


def generate_anotX_Shear(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, shear:float):
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


def generate_anotX_RandomShear(image_list:list, path_to_labels:str, path_to_save:str, name_id:str, shear:float):
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


def generate_anotX_flipud(image_list:list, path_to_labels:str, path_to_save:str, name_id:str):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = Rotate(180)(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)


def generate_anotX_fliplr(image_list:list, path_to_labels:str, path_to_save:str, name_id:str):
    for img_path in image_list:
        img_name = os.path.split(img_path)[1]
        img = cv2.imread(img_path)

        # Read XML
        xml_path = f"{path_to_labels}/{os.path.splitext(img_name)[0]}.xml"
        bbox_list, class_list = get_xml(xml_path)
        if len(bbox_list) > 0:
            # Augumentation
            bbox_list = np.array(bbox_list, dtype=np.float64)
            img_aug, bbox_aug = HorizontalFlip()(img.copy(), bbox_list.copy())
            if len(bbox_aug) > 0:
                # save image
                cv2.imwrite(f"{path_to_save}/{os.path.splitext(img_name)[0]}_{name_id}.jpg", img_aug)
                # Save XML
                xml_name = f"{os.path.splitext(img_name)[0]}_{name_id}.xml"
                path_to_xml_save = f"{path_to_save}/{xml_name}"
                write_xml(img_path, bbox_aug, class_list, path_to_xml_save)
