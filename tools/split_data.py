import os
import glob
import random
import shutil
import argparse
from tqdm import tqdm


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

allFiles = glob.glob(f"{args['image']}/*.jpg") + glob.glob(f"{args['image']}/*.png") + glob.glob(f"{args['image']}/*.jpeg")
random.shuffle(allFiles)
testNum = int(len(allFiles) * args['ratio'])
testFileLst = []
while True:
    ap = random.choice(allFiles)
    if ap not in testFileLst:
        testFileLst.append(ap)
        if len(testFileLst) == testNum:
            break

# Test Data
print(f"Test Data: {len(testFileLst)}/{len(allFiles)}")
for i in tqdm(range(len(testFileLst))):
    # for testFile in testFileLst:
    testFile = testFileLst[i]
    img_path, img_name = os.path.split(testFile)
    # image
    shutil.copy(testFile, f"{args['save']}/valid/images")
    # label
    name = os.path.splitext(img_name)[0]
    label_path = f"{args['label']}/{name}.{args['exe']}"
    shutil.copy(label_path, f"{args['save']}/valid/labels")
print("[COMPLETED] Test Data")

# Train Data
trainFileLst = list(set(testFileLst).symmetric_difference(set(allFiles)))
print(f"Train Data: {len(trainFileLst)}/{len(allFiles)}")
for i in tqdm(range(len(trainFileLst))):
    # for trainFile in trainFileLst:
    trainFile = trainFileLst[i]
    img_path, img_name = os.path.split(trainFile)
    # image
    shutil.copy(trainFile, f"{args['save']}/train/images")
    # label
    name = os.path.splitext(img_name)[0]
    label_path = f"{args['label']}/{name}.{args['exe']}"
    shutil.copy(label_path, f"{args['save']}/train/labels")
print("[COMPLETED] Train Data")
