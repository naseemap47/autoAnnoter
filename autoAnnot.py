from anot_utils import findBBox, save_xml, save_yolo, read_txt_lines
import cv2
import onnxruntime
import glob
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", action='store_true',
                help="to annotate in XML format")
ap.add_argument("-t", "--txt", action='store_true',
                help="to annotate in (.txt) format")
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to ONNX model")
ap.add_argument("-s", "--size", type=int, required=True,
                help="Size of image used to train the model")
ap.add_argument("-conf", "--confidence", type=float, required=True,
                help="Model detection Confidence (0<confidence<1)")
ap.add_argument("-r", "--remove", nargs='+', default=[],
                help="List of classes need to remove")
ap.add_argument("-k", "--keep", nargs='+', default=[],
                help="List of classes need to keep")
args = vars(ap.parse_args())

if len(args['remove'])>0 and len(args['keep'])>0:
    print('[INFO] use remove or keep NOT both...')

# ONNX Model
onnx_session = onnxruntime.InferenceSession(args['model'])

img_list = glob.glob(os.path.join(args["dataset"], '*.jpg')) + \
    glob.glob(os.path.join(args["dataset"], '*.jpeg')) + \
    glob.glob(os.path.join(args["dataset"], '*.png'))
class_names = read_txt_lines(args['classes'])

for img in img_list:
    image = cv2.imread(img)
    h, w, c = image.shape
    bbox_list, class_list, confidence = findBBox(
        onnx_session, image, args['size'], args['confidence'], class_names, args['remove'], args['keep'])
    folder_name, file_name = os.path.split(img)
    # XML Annotation
    if args['xml']:
        save_xml(folder_name, file_name, img, w, h, c,
                 bbox_list, class_list, class_names)
        print(f'Successfully Annotated {file_name}')

    # YOLO Annotation
    if args['txt']:
        save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
        print(f'Successfully Annotated {file_name}')
print(f"{'TXT' if args['txt'] else 'XML'}-Auto_Annotation Successfully Completed")
