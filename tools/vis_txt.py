import cv2
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-t", "--txt", type=str, required=True,
                help="path to dir/*.txt")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")
ap.add_argument("--save", action='store_true',
                help="Save image")
args = vars(ap.parse_args())

txt_name = args['txt']
path_to_class = args['classes']

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


img_file = cv2.imread(args['img'])
height_n, width_n, depth_n = img_file.shape
class_names = open(f'{path_to_class}', 'r+').read().splitlines()
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
txt_file = open(txt_name, 'r+')
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
    
    plot_one_box(bbox, img_file, label=class_names[int(class_index)], color=colors[int(class_index)], line_thickness=2)

# Save Image
if args['save']:
    cv2.imwrite('output.jpg', img_file)

cv2.imshow('img', img_file)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
