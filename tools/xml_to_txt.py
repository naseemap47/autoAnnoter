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

img_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

class_names = open(f'{path_to_class}', 'r+').read().splitlines()

for img in img_list:
    # image
    h, w, c = cv2.imread(img).shape
    path_to_dir, img_name = os.path.split(img)
    name = os.path.splitext(os.path.split(img)[1])[0]
    try:
        xml = f"{path_to_xml}/{name}.xml"
        bbox_list, class_list = get_xml(xml)
    except:
        print(f'[XML not found] {xml}')
        continue
    # TXT File
    txt_file_name = os.path.splitext(img_name)[0] + '.txt'
    path_to_save = os.path.join(path_to_dir, txt_file_name)
    out_file = open(path_to_save, 'w')
    if len(bbox_list)>0:
        for bbox, class_name in zip(bbox_list, class_list):
            try:
                class_selected_id = class_names.index(f'{class_name}')
                xmin, ymin, xmax, ymax = bbox
                x_center = float((xmin + xmax)) / 2 / w
                y_center = float((ymin + ymax)) / 2 / h
                width = float((xmax - xmin)) / w
                height = float((ymax - ymin)) / h
                
                # Save
                out_file.write("%d %.6f %.6f %.6f %.6f\n" %
                            (class_selected_id, x_center, y_center, width, height))
            except:
                print(f'Skipping: {class_name} not found')
        print(f'Successfully Created {txt_file_name}')
    else:
        print(f'[INFO] Empty Bounding Box in {xml}')
print('[INFO] Converted All XML Files to TXT Files')
