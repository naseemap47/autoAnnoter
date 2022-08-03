import os
import xml.etree.ElementTree as ET
from lxml import etree
from annoter import findClass
DEFAULT_ENCODING = 'utf-8'


# function to convert XML file
def save_xml(folder_name, file_name, path_txt, width_n, height_n, depth_n, obj_list, class_list):
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

        name.text = f'{findClass(class_id)}'
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
    xml_str = etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING).replace("  ".encode(), "\t".encode())
    
    path_to_dir = os.path.split(path_txt)[0]
    xml_file_name = os.path.splitext(file_name)[0] + '.xml'
    path_to_save = os.path.join(path_to_dir, xml_file_name)
    
    with open(f'{path_to_save}', 'w') as file:
        file.write(xml_str.decode('utf8'))

    print(f'Successfully Created {xml_file_name}')


if __name__=='__main__':

    data = ET.Element('annotation')

    folder = ET.SubElement(data, 'folder')


    filename = ET.SubElement(data, 'filename')
    path = ET.SubElement(data, 'path')
    sub = ET.SubElement(data, 'sub')
    sub_s = ET.SubElement(sub, 'subsub')


    # filename.set('type', 'Accepted')
    # path.set('type', 'Declined')


    folder.text = 'img'
    filename.text = "WhatsApp Image 2022-07-05 at 2.46.28 PM.jpeg"
    path.text = "/home/naseem/Downloads/img/WhatsApp Image 2022-07-05 at 2.46.28 PM.jpeg"
    sub_s.text = 'sub_sub'


    sample_xml = ET.tostring(data, 'utf8')
    root = etree.fromstring(sample_xml)

    xml_str = etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING).replace("  ".encode(), "\t".encode())

    with open('file.xml', 'w') as file:
        file.write(xml_str.decode('utf8'))

