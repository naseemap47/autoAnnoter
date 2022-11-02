import os
import xml.etree.ElementTree as ET
from lxml import etree
import argparse
import glob
import cv2
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")        
args = vars(ap.parse_args())

path_to_dir = args["dataset"]
path_to_save = args["save"]

# function to convert XML file
def save_xml(folder_name, file_name, path_txt, width_n, height_n, depth_n, path_to_save_dir):
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

    # Save
    sample_xml = ET.tostring(data, 'utf8')
    root = etree.fromstring(sample_xml)
    xml_str = etree.tostring(root, pretty_print=True, encoding='utf-8').replace(
        "  ".encode(), "\t".encode())

    # path_to_dir = os.path.split(path_txt)[0]
    xml_file_name = os.path.splitext(file_name)[0] + '.xml'
    path_to_save = os.path.join(path_to_save_dir, xml_file_name)

    with open(f'{path_to_save}', 'w') as file:
        file.write(xml_str.decode('utf8'))
    print(f'[INFO] Successfully Created {xml_file_name}')


# Create Save Dir If NOT Exist
os.makedirs(path_to_save, exist_ok=True)

# Images
img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
           glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
           glob.glob(os.path.join(path_to_dir, '*.png'))

for img in img_list:
    folder_name, file_name = os.path.split(img)
    image = cv2.imread(img)
    height, width, channel = image.shape
    save_xml(folder_name, file_name, img, width, height, channel, path_to_save)

    # Copy neg Image
    path_to_save_neg_img = os.path.join(path_to_save, file_name)
    shutil.copyfile(img, path_to_save_neg_img)
    print(f'[INFO] Successfully Saved {file_name}')

print('[INFO] Successfully Completed Neg-Image Annotation')
