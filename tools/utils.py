import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from lxml import etree
import cv2
import os


def get_xml(xml_path):
    bbox_temp_id = ['xmin', 'ymin', 'xmax', 'ymax']
    bbox_temp = [None, None, None, None]
    
    tree = ET.parse(xml_path)
    data = open(xml_path, 'r').read()
    bs_data = BeautifulSoup(data, "xml")
    b_unique = bs_data.find_all('object')
    count = len(b_unique)
    class_list = []
    bbox_list = []
    if count > 0:
        doc = etree.parse(xml_path)
        obj_pos = int(doc.xpath('count(//object[1]/preceding-sibling::*)'))
        bbox_pos_last = int(doc.xpath('count(//bndbox[1]/preceding-sibling::*)'))
        first_bbox_pos = int(bbox_pos_last/count)

        root = tree.getroot()
        bbox_temp = [None, None, None, None]
        for i in range(count):
            # class
            class_name = root[i+obj_pos][0].text
            # BBox
            bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][0].tag)] = int(root[i+obj_pos][first_bbox_pos][0].text)
            bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][1].tag)] = int(root[i+obj_pos][first_bbox_pos][1].text)
            bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][2].tag)] = int(root[i+obj_pos][first_bbox_pos][2].text)
            bbox_temp[bbox_temp_id.index(root[i+obj_pos][first_bbox_pos][3].tag)] = int(root[i+obj_pos][first_bbox_pos][3].text)

            xmin, ymin, xmax, ymax = bbox_temp
            bbox_list.append([xmin, ymin, xmax, ymax])
            class_list.append(class_name)
    
    return bbox_list, class_list


def write_xml(img_path, bbox_list, class_list, path_to_save_with_name):
    path_to_dir, img_name = os.path.split(img_path)
    img_file = cv2.imread(img_path)
    h, w, c = img_file.shape
    data = ET.Element('annotation')
    folder = ET.SubElement(data, 'folder')
    filename = ET.SubElement(data, 'filename')
    path = ET.SubElement(data, 'path')

    folder.text = f'{path_to_dir}'
    filename.text = f"{img_name}"
    path.text = f"{img_path}"

    source = ET.SubElement(data, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(data, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')

    width.text = f'{w}'
    height.text = f'{h}'
    depth.text = f'{c}'

    segmented = ET.SubElement(data, 'segmented')
    segmented.text = '0'

    # Object
    if len(bbox_list) > 0 and len(class_list) > 0:
        for obj, class_name in zip(bbox_list, class_list):
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

            name.text = f'{class_name}'
            pose.text = 'Unspecified'
            truncated.text = '0'
            difficult.text = '0'

            xmin.text = f'{int(obj[0])}'
            ymin.text = f'{int(obj[1])}'
            xmax.text = f'{int(obj[2])}'
            ymax.text = f'{int(obj[3])}'

    # Save
    sample_xml = ET.tostring(data, 'utf8')
    root = etree.fromstring(sample_xml)
    xml_str = etree.tostring(root, pretty_print=True, encoding='utf-8').replace(
        "  ".encode(), "\t".encode())

    with open(path_to_save_with_name, 'w') as file:
        file.write(xml_str.decode('utf8'))

    print(f"[INFO] Saved: {path_to_save_with_name}")


def get_txt(path_to_txt, path_to_img):
    img_file = cv2.imread(path_to_img)
    height_n, width_n, depth_n = img_file.shape
    
    txt_file = open(path_to_txt, 'r+')
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
    
    return obj_list, class_list
