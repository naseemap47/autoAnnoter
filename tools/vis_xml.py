import cv2
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from lxml import etree
import cv2
import argparse
import random


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-x", "--xml", type=str, required=True,
                help="path to dir/*.xml")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")
ap.add_argument("--save", action='store_true',
                help="Save image")
args = vars(ap.parse_args())


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class_names = open(f"{args['classes']}", 'r+').read().splitlines()
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

tree = ET.parse(args['xml'])
data = open(args['xml'], 'r').read()
bs_data = BeautifulSoup(data, "xml")
b_unique = bs_data.find_all('object')
count = len(b_unique)

doc = etree.XML(data)
obj_pos = int(doc.xpath('count(//object[1]/preceding-sibling::*)'))
bbox_pos_last = int(doc.xpath('count(//bndbox[1]/preceding-sibling::*)'))
first_bbox_pos = int(bbox_pos_last/count)

root = tree.getroot()
path = root[2].text

img_file = cv2.imread(args['img'])
h, w, c = img_file.shape
txt_list = []
bbox_list = []
bbox_temp_id = ['xmin', 'ymin', 'xmax', 'ymax']
bbox_temp = [None, None, None, None]

for i in range(count):
    bbox_temp = [None, None, None, None]

    class_name = root[i+obj_pos][0].text
    class_selected_id = class_names.index(f'{class_name}')
    
    bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][0].tag)] = int(root[i+obj_pos][first_bbox_pos][0].text)
    bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][1].tag)] = int(root[i+obj_pos][first_bbox_pos][1].text)
    bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][2].tag)] = int(root[i+obj_pos][first_bbox_pos][2].text)
    bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][3].tag)] = int(root[i+obj_pos][first_bbox_pos][3].text)

    plot_one_box(bbox_temp, img_file, colors[class_selected_id], class_name)

# Save Image
if args['save']:
    cv2.imwrite('output.jpg', img_file)

cv2.imshow('img', img_file)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
