import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from lxml import etree
DEFAULT_ENCODING = 'utf-8'


# Read Classes.txt
def read_txt_lines(path_to_txt):
    with open(path_to_txt, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


# Conver into Class Names
def findClass(class_id, class_names):
    class_name = class_names[int(class_id)-1]
    return class_name

# Remove Class from Annotaion
def remove_class(class_list, id, remove_list):
    name = class_list[id]
    return name in set(remove_list)

# to get bounding box from ONNX model
def findBBox(onnx_session, img, img_resize, threshold, class_name_list, remove_list):
  # onnx session
    input_name = onnx_session.get_inputs()[0].name

    # Image
    h, w, c = img.shape
    img_resized = cv2.resize(img, (img_resize, img_resize))
    img_data = np.reshape(img_resized, (1, img_resize, img_resize, 3))
    img_data = img_data.astype('uint8')
    ort_inputs = {input_name: img_data}
    ort_outs = onnx_session.run(None, ort_inputs)
    bbox_list = []
    class_list = []
    confidence = []
    c = 0
    for i in ort_outs[4][0]:
        if i > threshold:
            if not remove_class(class_name_list, int(ort_outs[2][0][c]), remove_list):
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

        c += 1
    return bbox_list, class_list, confidence


# function to convert XML file
def save_xml(folder_name, file_name, path_txt, width_n, height_n, depth_n, obj_list, class_list, class_names):
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

        name.text = f'{findClass(class_id, class_names)}'
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
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]

        x_center = float((x_min + x_max)) / 2 / w
        y_center = float((y_min + y_max)) / 2 / h

        width = float((x_max - x_min)) / w
        height = float((y_max - y_min)) / h

        # Save
        out_file.write("%d %.6f %.6f %.6f %.6f\n" %
                       (int(class_index-1), x_center, y_center, width, height))

    print(f'Successfully Created {txt_name}')


# YOLOv7
def get_BBoxYOLOv7(img, yolo_model, detect_conf, class_name_list, remove_list):

    # Load YOLOv7 model on Image
    results = yolo_model(img)

    # Bounding Box
    box = results.pandas().xyxy[0]
    bbox_list = []
    confidence = []
    class_ids = []
    # Class
    class_list = box['class'].tolist()
    # save_yolo function need class index starting from 1 NOT Zero
    new_list = [x+1 for x in class_list]

    for i, id in zip(box.index, new_list):
        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][i]

        # detect_conf
        if conf > detect_conf:
            # Remove specfic classes from Annotation
            if not remove_class(class_name_list, int(id-1), remove_list):
                # BBox
                bbox_list.append([xmin, ymin, xmax, ymax])
                # class
                class_ids.append(id)
                # Confidence
                confidence.append(conf)
    return bbox_list, class_ids, confidence


# YOLOv8
def get_BBoxYOLOv8(img, yolo_model, detect_conf, class_name_list, remove_list):

    bbox_list = []
    confidence = []
    class_ids = []

    # Load YOLOv8 model on Image
    results = yolo_model(img)

    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls
        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            # detect_conf
            if cnf > detect_conf:
                # Remove specfic classes from Annotation
                if not remove_class(class_name_list, int(cs), remove_list):
                    # BBox
                    bbox_list.append([xmin, ymin, xmax, ymax])
                    # class
                    class_ids.append(int(cs+1))
                    # Confidence
                    confidence.append(cnf)

    return bbox_list, class_ids, confidence


# YOLOv8
def get_BBoxYOLONAS(img, yolo_model, detect_conf, remove_list):

    bbox_list = []
    confidence = []
    class_ids = []

    # Load YOLOv8 model on Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = next(yolo_model.predict(img_rgb)._images_prediction_lst)
    class_name_list = preds.class_names
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    for box, cnf, cs in zip(bboxes, confs, labels):
        # detect_conf
        if cnf > detect_conf:
            # Remove specfic classes from Annotation
            if not remove_class(class_name_list, int(cs), remove_list):
                # BBox
                bbox_list.append([box[:4]])
                # class
                class_ids.append(int(cs+1))
                # Confidence
                confidence.append(cnf)

    return bbox_list, class_ids, confidence
