import os
import glob
import random
import shutil
import argparse
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-s", "--save", type=str, required=True,
                help="path to save")
ap.add_argument("-r", "--ratio", type=float, default=0.2,
                help="test ratio")
args = vars(ap.parse_args())

class_list = sorted(os.listdir(args['image']))
print("Class List: ", class_list)

for class_name in class_list:
    allFiles = glob.glob(f"{args['image']}/{class_name}/*.jpg")
    random.shuffle(allFiles)
    testNum = int(len(allFiles) * args['ratio'])
    if testNum == 0:
        print(f"[WARNING] Less Data: {class_name}")
        continue
    os.makedirs(f"{args['save']}/train/{class_name}", exist_ok=True)
    os.makedirs(f"{args['save']}/val/{class_name}", exist_ok=True)
    # print(testNum)
    testFileLst = []
    while True:
        ap = random.choice(allFiles)
        if ap not in testFileLst:
            testFileLst.append(ap)
            if len(testFileLst) == testNum:
                break

    print(f"[INFO] Class: {class_name}")
    # Test Data
    print(f"Test Data: {len(testFileLst)}/{len(allFiles)}")
    for i in tqdm(range(len(testFileLst))):
        testFile = testFileLst[i]
        shutil.copy(testFile, f"{args['save']}/val/{class_name}")

    trainFileLst = list(set(testFileLst).symmetric_difference(set(allFiles)))
    print(f"Train Data: {len(trainFileLst)}/{len(allFiles)}")
    for i in tqdm(range(len(trainFileLst))):
        trainFile = trainFileLst[i]
        shutil.copy(trainFile, f"{args['save']}/train/{class_name}")
    print(f"[COMPLETED] Class: {class_name}")
