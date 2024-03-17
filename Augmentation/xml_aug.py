from Augmentation.aug_utils import *
from numpy.random import choice
import glob
import os
import argparse
import yaml


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-x", "--xml", type=str, required=True,
                help="path to xml/dir")
ap.add_argument("-s", "--save", type=str, required=True,
                help="path to save aug xml")
ap.add_argument("-y", "--yaml", type=str, default='Augmentation/default.yaml',
                help="path to aug yaml file")
args = vars(ap.parse_args())


with open(args['yaml'], 'r') as f:
	data = yaml.load(f, Loader=yaml.SafeLoader)
print('Augmentation Params:\n', data)

os.makedirs(args['save'], exist_ok=True)
img_full_list = glob.glob(f"{args['image']}/*.jpeg") + \
                glob.glob(f"{args['image']}/*.jpg")  + \
                glob.glob(f"{args['image']}/*.png")
img_list = sorted(img_full_list)

# image HSV-Hue augmentation
total_prob = int((data['hsv_h']['prob'])*len(img_list))
hsv_h_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_h_img, args['xml'], hue=data['hsv_h']['hue'], path_to_save=args['save'], name_id='hsv_h')

# image HSV-Saturation augmentation
total_prob = int((data['hsv_s']['prob'])*len(img_list))
hsv_s_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_s_img, args['xml'], saturation=data['hsv_s']['saturation'], path_to_save=args['save'], name_id='hsv_s')

# image HSV-Value (brightness) augmentation
total_prob = int((data['hsv_v']['prob'])*len(img_list))
hsv_v_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(hsv_v_img, args['xml'], brightness=data['hsv_v']['brightness'], path_to_save=args['save'], name_id='hsv_v')

# Mixed image HSV augmentation (Mixed HSV-Hue, HSV-Saturation and HSV-Value (brightness))
total_prob = int((data['hsv']['prob'])*len(img_list))
hsv_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomHSV(
    hsv_img, args['xml'], hue=data['hsv']['hue'], saturation=data['hsv']['saturation'], 
    brightness=data['hsv']['brightness'], path_to_save=args['save'], name_id='hsv'
)

# image rotation augmentation
total_prob = int((data['degrees']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Rotate(rot_img, args['xml'], args['save'], 'rot', data['degrees']['deg'])

# image Random rotation augmentation
total_prob = int((data['degrees_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomRotate(rot_img, args['xml'], args['save'], 'rot_random', data['degrees_random']['deg'])

# image Translate augmentation
total_prob = int((data['translate']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Translate(rot_img, args['xml'], args['save'], 'trans', data['translate']['translate_x'], data['translate']['translate_y'])

# image Random Translate augmentation
total_prob = int((data['translate_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomTranslate(rot_img, args['xml'], args['save'], 'trans_random', data['translate_random']['translate'])

# image Scale augmentation
total_prob = int((data['scale']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Scale(rot_img, args['xml'], args['save'], 'scale', data['scale']['scale_x'], data['scale']['scale_y'])

# image Random Scale augmentation
total_prob = int((data['scale_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomScale(rot_img, args['xml'], args['save'], 'scale_random', data['scale_random']['scale'])

# image Shear augmentation
total_prob = int((data['shear']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_Shear(rot_img, args['xml'], args['save'], 'shear', data['shear']['shear'])

# image Random Shear augmentation
total_prob = int((data['shear_random']['prob'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_RandomShear(rot_img, args['xml'], args['save'], 'shear_random', data['shear_random']['shear'])

# image flip up-down augmentation
total_prob = int((data['flipud'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_flipud(rot_img, args['xml'], args['save'], 'flipud')

# image flip left-right augmentation
total_prob = int((data['fliplr'])*len(img_list))
rot_img = choice(img_list, total_prob, replace=False)
generate_anot_fliplr(rot_img, args['xml'], args['save'], 'fliplr')

