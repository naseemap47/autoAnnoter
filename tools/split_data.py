import os
import glob
import random
import shutil
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-l", "--label", type=str, required=True,
                help="path to txt/dir")
ap.add_argument("-s", "--save", type=str, required=True,
                help="path to save")
ap.add_argument("-e", "--exe", type=str, default='txt',
                choices=['txt', 'xml'],
                help="label exe to identify labels")
ap.add_argument("-r", "--ratio", type=float, default=0.2,
                help="test ratio")
args = vars(ap.parse_args())

os.makedirs(f"{args['save']}/train", exist_ok=True)
os.makedirs(f"{args['save']}/valid", exist_ok=True)
os.makedirs(f"{args['save']}/train/images", exist_ok=True)
os.makedirs(f"{args['save']}/valid/images", exist_ok=True)
os.makedirs(f"{args['save']}/train/labels", exist_ok=True)
os.makedirs(f"{args['save']}/valid/labels", exist_ok=True)

allFiles = glob.glob(f"{args['image']}/*.jpg")

testNum = int(len(allFiles) * args['ratio'])
testFileLst = []
while True:
    ap = random.choice(allFiles)
    if ap not in testFileLst:
        testFileLst.append(ap)
        if len(testFileLst) == testNum:
            break

trainFileLst = list(set(testFileLst).symmetric_difference(set(allFiles)))

for testFile in testFileLst:
    img_path, img_name = os.path.split(testFile)
    # image
    shutil.copy(testFile, f"{args['save']}/valid/images")
    # label
    name = os.path.splitext(img_name)[0]
    label_path = f"{args['label']}/{name}.{args['exe']}"
    shutil.copy(label_path, f"{args['save']}/valid/labels")

for trainFile in trainFileLst:
    img_path, img_name = os.path.split(trainFile)
    # image
    shutil.copy(trainFile, f"{args['save']}/train/images")
    # label
    name = os.path.splitext(img_name)[0]
    label_path = f"{args['label']}/{name}.{args['exe']}"
    shutil.copy(label_path, f"{args['save']}/train/labels")
