import cv2
from annoter import findBBox
import argparse
import glob
import os
from make_xml import save_xml



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to ONNX model")
args = vars(ap.parse_args())
path_to_dir = args["dataset"]
onnx_model_path = args['model']

img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
           glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
           glob.glob(os.path.join(path_to_dir, '*.png'))

# print(img_list)

for img in img_list:
    image = cv2.imread(img)
    h, w, c = image.shape
    bbox_list, class_list, confidence = findBBox(onnx_model_path, image, 320, 0.4)
    folder_name, file_name = os.path.split(img)
    save_xml(folder_name, file_name, img, w, h, c, bbox_list, class_list)
    print(f'Successfully Annotated {file_name}')

print('Auto-Annotation Successfully Completed')

