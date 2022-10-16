import glob
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from lxml import etree
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-x", "--xml", type=str, required=True,
                help="path to xml/dir")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_to_xml = args['xml']
path_to_class = args['classes']

xml_list = sorted(glob.glob(f'{path_to_xml}/*.xml'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

img_list = sorted(img_full_list)
class_names = open(f'{path_to_class}', 'r+').read().splitlines()

for xml, img in zip(xml_list, img_list):
    path_to_dir, img_name = os.path.split(img)
    tree = ET.parse(xml)
    with open(xml, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    b_unique = bs_data.find_all('object')

    root = tree.getroot()
    path = root[2].text
    count = len(b_unique)
    img_file = cv2.imread(img)
    h, w, c = img_file.shape
    txt_list = []
    bbox_list = []
    for i in range(count):
        class_name = root[i+6][0].text
        class_selected_id = class_names.index(f'{class_name}')
        
        xmin = int(root[i+6][4][0].text)
        ymin = int(root[i+6][4][1].text)
        xmax = int(root[i+6][4][2].text)
        ymax = int(root[i+6][4][3].text)
    
        x_center = float((xmin + xmax)) / 2 / w
        y_center = float((ymin + ymax)) / 2 / h
        width = float((xmax - xmin)) / w
        height = float((ymax - ymin)) / h

        bbox_list.append([class_selected_id, x_center, y_center, width, height])

    # TXT File
    if len(bbox_list)>0:
        txt_file_name = os.path.splitext(img_name)[0] + '.txt'
        path_to_save = os.path.join(path_to_dir, txt_file_name)
        out_file = open(path_to_save, 'w')
        for bbox in bbox_list:
            # Save
            out_file.write("%d %.6f %.6f %.6f %.6f\n" %
                        (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))

        print(f'Successfully Created {txt_file_name}')
    else:
        print(f'[INFO] Empty Bounding Box in {xml}')
print('[INFO] Converted All XML Files to TXT Files')
