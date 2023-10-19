import glob
import os
import cv2
import argparse
from tool_utils import get_xml


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
    # image
    h, w, c = cv2.imread(img).shape
    path_to_dir, img_name = os.path.split(img)
    bbox_list, class_list = get_xml(xml)
    
    # TXT File
    txt_file_name = os.path.splitext(img_name)[0] + '.txt'
    path_to_save = os.path.join(path_to_dir, txt_file_name)
    out_file = open(path_to_save, 'w')
    if len(bbox_list)>0:
        for bbox, class_name in zip(bbox_list, class_list):
            class_selected_id = class_names.index(f'{class_name}')
            xmin, ymin, xmax, ymax = bbox
            x_center = float((xmin + xmax)) / 2 / w
            y_center = float((ymin + ymax)) / 2 / h
            width = float((xmax - xmin)) / w
            height = float((ymax - ymin)) / h
            
            # Save
            out_file.write("%d %.6f %.6f %.6f %.6f\n" %
                        (class_selected_id, x_center, y_center, width, height))

        print(f'Successfully Created {txt_file_name}')
    else:
        print(f'[INFO] Empty Bounding Box in {xml}')
print('[INFO] Converted All XML Files to TXT Files')
