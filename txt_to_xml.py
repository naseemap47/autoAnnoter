import glob
import os
import xml.etree.ElementTree as ET
from lxml import etree
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

DEFAULT_ENCODING = 'utf-8'
txt_list = sorted(glob.glob(f'{path_to_txt}/*.txt'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

img_list = sorted(img_full_list)
class_names = open(f'{path_to_class}', 'r+').read().splitlines()

for txt, img in zip(txt_list, img_list):
    # Locations
    folder_name, file_name = os.path.split(img)
    path_txt = img
    img_file = cv2.imread(img)
    height_n, width_n, depth_n = img_file.shape
    
    txt_file = open(txt, 'r+')
    lines = txt_file.read().splitlines()
    obj_list = []
    class_list = []
    for line in lines:
        class_index, x_center, y_center, width, height = line.split()
        xmax = int((float(x_center)*width_n) + (float(width) * width_n)/2.0)
        xmin = int((float(x_center)*width_n) - (float(width) * width_n)/2.0)
        ymax = int((float(y_center)*height_n) + (float(height) * height_n)/2.0)
        ymin = int((float(y_center)*height_n) - (float(height) * height_n)/2.0)
        bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]

        obj_list.append(bbox)
        class_list.append(int(class_index))

    if len(class_list)>0:
        data = ET.Element('annotation')
        folder = ET.SubElement(data, 'folder')
        filename = ET.SubElement(data, 'filename')
        path = ET.SubElement(data, 'path')

        folder.text = f'{folder_name}'
        filename.text = f"{file_name}"
        path.text = f"{path_txt}"

        source = ET.SubElement(data, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        size = ET.SubElement(data, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')

        width.text = f'{width_n}'
        height.text = f'{height_n}'
        depth.text = f'{depth_n}'

        segmented = ET.SubElement(data, 'segmented')
        segmented.text = '0'

        # Object
        for obj, class_id in zip(obj_list, class_list):
            object = ET.SubElement(data, 'object')
            name = ET.SubElement(object, 'name')
            pose = ET.SubElement(object, 'pose')
            truncated = ET.SubElement(object, 'truncated')
            difficult = ET.SubElement(object, 'difficult')
            bndbox = ET.SubElement(object, 'bndbox')

            # BBox
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')

            name.text = f'{class_names[class_id]}'
            pose.text = 'Unspecified'
            truncated.text = '0'
            difficult.text = '0'

            xmin.text = f'{obj[0]}'
            ymin.text = f'{obj[1]}'
            xmax.text = f'{obj[2]}'
            ymax.text = f'{obj[3]}'

        # Save
        sample_xml = ET.tostring(data, 'utf8')
        root = etree.fromstring(sample_xml)
        xml_str = etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING).replace(
            "  ".encode(), "\t".encode())

        # path_to_dir = os.path.split(path_to_img)[0]
        xml_file_name = os.path.splitext(file_name)[0] + '.xml'
        path_to_save = os.path.join(path_to_img, xml_file_name)

        with open(f'{path_to_save}', 'w') as file:
            file.write(xml_str.decode('utf8'))

        print(f'Successfully Created {xml_file_name}')
    
    else:
        print(f'[INFO] Annotation {txt} File is Empty')
