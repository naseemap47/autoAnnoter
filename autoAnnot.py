from utils import findBBox, save_xml, save_yolo
import cv2
import glob
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", action='store_true',
                help="to annotate in XML format")
ap.add_argument("-y", "--yolo", action='store_true',
                help="to annotate in YOLO format(.txt)")
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to ONNX model")
ap.add_argument("-s", "--size", type=int, required=True,
                help="Size of image used to train the model")
ap.add_argument("-c", "--confidence", type=float, required=True,
                help="Model detection Confidence (0<confidence<1)")


args = vars(ap.parse_args())
XML = args['xml']
YOLO = args['yolo']
path_to_dir = args["dataset"]
onnx_model_path = args['model']
img_size = args['size']
detect_conf = args['confidence']


img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
    glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
    glob.glob(os.path.join(path_to_dir, '*.png'))

# XML Annotation
if XML:
    for img in img_list:
        image = cv2.imread(img)
        h, w, c = image.shape
        bbox_list, class_list, confidence = findBBox(
            onnx_model_path, image, img_size, detect_conf)
        folder_name, file_name = os.path.split(img)
        save_xml(folder_name, file_name, img, w, h, c, bbox_list, class_list)
        print(f'Successfully Annotated {file_name}')

    print('XML-Auto_Annotation Successfully Completed')

# YOLO Annotation
if YOLO:
    for img in img_list:
        image = cv2.imread(img)
        h, w, c = image.shape
        bbox_list, class_list, confidence = findBBox(
            onnx_model_path, image, img_size, detect_conf)
        folder_name, file_name = os.path.split(img)
        save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
        print(f'Successfully Annotated {file_name}')

    print('YOLO-Auto_Annotation Successfully Completed')
