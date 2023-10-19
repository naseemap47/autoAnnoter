from data_aug import RandomHorizontalFlip, RandomScale, RandomTranslate, \
    RandomRotate, RandomShear, Resize, RandomHSV, Sequence, Rotate
import glob
import os
import cv2
import argparse
from tool_utils import get_txt, write_txt
import numpy as np
from bbox_util import rotate_im


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-t", "--txt", type=str, required=True,
                help="path to txt/dir")
ap.add_argument("-s", "--save", type=str, required=True,
                help="path to save")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_to_txt = args['txt']
path_to_save = args['save']
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(f"{path_to_save}/images", exist_ok=True)
os.makedirs(f"{path_to_save}/labels", exist_ok=True)

txt_list = sorted(glob.glob(f'{path_to_txt}/*.txt'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

img_list = sorted(img_full_list)
# img_aug = np.zeros((2, 2, 3))
# img_aug = None
# c = 0
for txt_path, img_path in zip(txt_list, img_list):
    # c += 1
    # Locations
    folder_name, img_name = os.path.split(img_path)
    bbox_list, class_list = get_txt(txt_path, img_path)
    img = cv2.imread(img_path)
    
    if len(bbox_list) > 0:
        # Augumentation
        bbox_list = np.array(bbox_list, dtype=np.float64)
        # if c%3 == 0:
        img_aug, bbox_aug = Rotate(180)(img.copy(), bbox_list.copy())
    else:
        # img_aug = img[:, ::-1, :]
        img_aug = rotate_im(img, 180)
        bbox_aug = []

        # elif c%4 == 0:
        #     img_aug, bbox_aug = RandomScale(0.2, diff = True)(img.copy(), bbox_list.copy())
        # elif c%5 == 0:
        #     img_aug, bbox_aug = RandomRotate(20)(img.copy(), bbox_list.copy())
        # elif c%6 == 0:
        #     img_aug, bbox_aug = RandomShear(0.2)(img.copy(), bbox_list.copy())
        # else:

        # Write Aug
        # if len(bbox_aug)>0:
    h, w, _ = img_aug.shape
    cv2.imwrite(f"{path_to_save}/images/{img_name}", img_aug)
    anot_name = f"{path_to_save}/labels/{os.path.splitext(img_name)[0]}.txt"
    write_txt(anot_name, bbox_aug, class_list, h, w)
    print(f"[INFO] Saved {anot_name}")
