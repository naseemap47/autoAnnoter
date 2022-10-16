import glob
import os
import xml.etree.ElementTree as et
from bs4 import BeautifulSoup
import shutil
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")
ap.add_argument("-n", "--name", type=str, required=True,
                help="name of class, that wants filter from pascal voc annotaion XML file")
                
args = vars(ap.parse_args())

path_to_dir = args["dataset"]
path_to_save = args["save"]
select_class_name = args['name']


def find_pothole_label_in_xml(path_to_xml, save_dir):
    # print(f'Opened {os.path.split(path_to_xml)[1]}')
    tree = et.parse(path_to_xml)
    root = tree.getroot()

    with open(path_to_xml, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    b_unique = bs_data.find_all('object')

    count = len(b_unique)
    # print(b_unique)
    filename = root[1].text
    # print('filename: ', filename)
    xml_filename = os.path.splitext(filename)[0] + '.xml'

    for i in range(count):
        position_id = -(i+1)
        class_name = root[position_id][0].text
        if class_name==f'{select_class_name}':
            print(f'Founded {select_class_name.upper()} in XML file')
            shutil.move(os.path.join(path_to_dir, filename), os.path.join(save_dir, filename))
            print(f'Moved {filename} file')
            shutil.move(path_to_xml, os.path.join(save_dir, xml_filename))
            print(f'Moved {xml_filename} file')
            break
        else:
            print(f'NOT found {select_class_name.upper()} labe in {filename}')



xml_list = glob.glob(os.path.join(path_to_dir, '*.xml'))
for xml in xml_list:
    find_pothole_label_in_xml(xml, path_to_save)
print(f'Find {select_class_name.upper()} Label in XML file - Task is Completed...')
