import cv2
import onnxruntime
import numpy as np
import os
import xml.etree.ElementTree as ET
from lxml import etree
from annoter import findClass
DEFAULT_ENCODING = 'utf-8'


# to get bounding box from ONNX model
def findBBox(onnx_model_path, img, img_resize, threshold):
  # Load Saved ONNX model
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # Image
    h, w, c = img.shape
    img_resized = cv2.resize(img, (img_resize, img_resize))
    img_data = np.reshape(img_resized, (1, img_resize, img_resize, 3))
    img_data = img_data.astype('uint8')
    ort_inputs = {input_name: img_data}
    ort_outs = session.run(None, ort_inputs)
    bbox_list = []
    class_list = []
    confidence = []
    c = 0
    for i in ort_outs[4][0]:
        if i > threshold:
            bbox = ort_outs[1][0][c]
            ymin = (bbox[0])
            xmin = (bbox[1])
            ymax = (bbox[2])
            xmax = (bbox[3])
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            xmin, ymin, xmax, ymax = int(
                xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)
            bbox_list.append([xmin, ymin, xmax, ymax])

            # Detection Classes
            class_list.append(ort_outs[2][0][c])

            # confidence
            confidence.append(i)

        c = c + 1
    return bbox_list, class_list, confidence


# Conver into Class Names
def findClass(class_id):
    if class_id == 1:
        class_name = 'pothole'
    else:
        class_name = 'patch'
    return class_name


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
    xml_str = etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING).replace(
        "  ".encode(), "\t".encode())

    path_to_dir = os.path.split(path_txt)[0]
    xml_file_name = os.path.splitext(file_name)[0] + '.xml'
    path_to_save = os.path.join(path_to_dir, xml_file_name)

    with open(f'{path_to_save}', 'w') as file:
        file.write(xml_str.decode('utf8'))

    print(f'Successfully Created {xml_file_name}')


# Function to convert YOLO (.txt) format
def save_yolo(folder_name, file_name, w, h, bbox_list, class_list):
    txt_name = os.path.splitext(file_name)[0] + '.txt'
    path_to_save = os.path.join(folder_name, txt_name)
    out_file = open(path_to_save, 'w')
    for box, class_index in zip(bbox_list, class_list):
        x_min = box[0]
        x_max = box[1]
        y_min = box[2]
        y_max = box[3]

        x_center = float((x_min + x_max)) / 2 / w
        y_center = float((y_min + y_max)) / 2 / h

        width = float((x_max - x_min)) / w
        height = float((y_max - y_min)) / h

        # Save
        out_file.write("%d %.6f %.6f %.6f %.6f\n" %
                       (int(class_index)-1, x_center, y_center, width, height))

    print(f'Successfully Created {txt_name}')
