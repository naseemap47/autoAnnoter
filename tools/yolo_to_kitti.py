import argparse
import os
import cv2
import glob


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-t", "--txt", type=str, required=True,
                help="path to txt/dir")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_to_txt = args['txt']
path_to_class = args['classes']

os.makedirs("KITTI_"+path_to_txt, exist_ok=True)
txt_list = sorted(glob.glob(f'{path_to_txt}/*.txt'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

img_list = sorted(img_full_list)
class_names = open(f'{path_to_class}', 'r+').read().splitlines()
for txt, img in zip(txt_list, img_list):
    img_file = cv2.imread(img)
    image_height, image_width, _ = img_file.shape

    with open(txt, 'r') as fh:
        data=fh.readlines()

    for cc, lines in enumerate(data):
        lines = lines.replace('\n', '').split(" ")
        lines[0] = int(lines[0])
        lines[1] = float(lines[1])*image_width
        lines[2] = float(lines[2])*image_height
        lines[3] = float(lines[3])*image_width
        lines[4] = float(lines[4])*image_height

        lines[0] = class_names[lines[0]]
        n_line = [0] * 15
        n_line[0] = lines[0]
        n_line[4] = int(lines[1]-lines[3]/2)
        n_line[5] = int(lines[2]-lines[4]/2)
        n_line[6] = int(lines[1]+lines[3]/2)
        n_line[7] = int(lines[2]+lines[4]/2)

        for c, n in enumerate(n_line):
            if not c == 0:
                if n < 0:
                    n_line[c] = 1
                if n > 256:
                    n_line[c] = 255

        for cc2, char in enumerate(n_line):
            if cc2 == 1:
                n_line[cc2] = float(0)
            if cc2 > 7:
                n_line[cc2] = 1

        str1 = ' '.join(str(n_line)).replace(' ', '').replace(',', ' ').replace('[', '').replace("]","").replace("'","").replace('"',"")
        data[cc] = str1+"\n"
    
    strf = " ".join(str(x) for x in data)
    strf = strf.replace("\n ","\n")
    file = open("KITTI_"+txt, "w")
    file.write(strf)
    file.close()
    print(f"[INFO] Completed {txt}")
