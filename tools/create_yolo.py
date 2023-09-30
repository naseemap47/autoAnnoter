import glob
import shutil
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to image/dir/data")
args = vars(ap.parse_args())


def move_file(f_list, dst):
    for f in f_list:
        shutil.move(f, dst)
        print(f"[INFO] Successfully Moved {os.path.split(f)[1]}")


img_list = glob.glob(os.path.join(args["data"], '*.jpg')) + \
    glob.glob(os.path.join(args["data"], '*.jpeg')) + \
    glob.glob(os.path.join(args["data"], '*.png'))

if len(img_list)>0:
    os.makedirs(os.path.join(args["data"], 'images'), exist_ok=True)

txt_list = glob.glob(os.path.join(args["data"], '*.txt'))
xml_list = glob.glob(os.path.join(args["data"], '*.xml'))

if len(txt_list)>0 and len(xml_list) == 0:
    os.makedirs(os.path.join(args["data"], 'labels'), exist_ok=True)
    move_file(img_list, os.path.join(args["data"], 'images'))
    move_file(txt_list, os.path.join(args["data"], 'labels'))
    
elif len(txt_list) == 0 and len(xml_list)>0:
    os.makedirs(os.path.join(args["data"], 'labels'), exist_ok=True)
    move_file(img_list, os.path.join(args["data"], 'images'))
    move_file(xml_list, os.path.join(args["data"], 'labels'))

elif len(txt_list)>0 and len(xml_list)>0:
    os.makedirs(os.path.join(args["data"], 'txt_labels'), exist_ok=True)
    os.makedirs(os.path.join(args["data"], 'xml_labels'), exist_ok=True)
    move_file(img_list, os.path.join(args["data"], 'images'))
    move_file(txt_list, os.path.join(args["data"], 'txt_labels'))
    move_file(xml_list, os.path.join(args["data"], 'xml_labels'))
    
else:
    print('[INFO] Labels NOT Found!!!')

