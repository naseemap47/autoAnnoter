# autoAnnoter
autoAnnoter its a tool to auto annotate data using a exisiting model

## 🚀 New Update (01-06-2023)
### Auto Annotate using YOLO-NAS Model 🥳
We can auto annotate data using new YOLO-NAS model. 

## Updates
- (**25-04-2023**): We can remove any classes from auto annotation. So that we can create new set of dataset using existing model.
  Example:
  - If we need to create a people detection model. We can create new dataset from existing COCO model.
  - If I need to create a dataset to train a model to detect **helmet**, **shoe** and **person**. But I can't annotate big dataset. But I have a model to detect **Helmet**, **Person** and other classes as well. But I can use this new feature (**Remove Classes from Auto Annotation**) only annotate **Helmet** and **Person**. This will reduce **2/3 of my work**. After this I only need to annotate **Shoe**.

- (**24-04-2023**): Added visualization tool and fixed issue with **XML** to **TXT** conversion.
  - **xml_to_txt.py** (Updated)
    - Fixed issue with, when unexpected format of xml annotation came, position of **xmax** and **ymin** maybe change, now it can handle any bounding box format.
  - Added XML and TXT annotation visualization tool
    - **vis_xml.py** : to visulise **xml (Pascal VOC)** annotation format
    - **vis_txt.py** : to visulise **txt (YOLO)** annotation format

- (**03-02-2023**) **YOLOv8** Auto Annotation
  - Auto Annotate using YOLOv8 Model

 - (**02-11-2022**) Added tools to this repository, that can help you to setup your Dataset for the Training.
    - **partition_dataset.py** : Partition your Dataset (**XML & TXT & Images**) into Train and Test in ratio
    - **txt_to_xml.py** : Convert your **TXT** Annotation files into **XML** Format
    - **xml_to_txt.py** : Convert your **XML** Annotation files into **TXT** Format
    - **xml_neg_annotation.py** : Annoatate your **Negative Dataset**
    - **find_oneClass_from_xml.py** : To filter your **PASCAL VOC** annotaion **XML** file based on **class name**

## Clone this GitHub Repository
```
git clone https://github.com/naseemap47/autoAnnoter.git
```
## Install Dependencies
```
cd autoAnnoter/
pip3 install -r requirements.txt
```
## 1. ONNX Model

<details>
  <summary>Args</summary>
  
  `-x`, `--xml`: to annotate in XML format <br>
  `-t`, `--txt`: to annotate in (.txt) format <br>
  `-i`, `--dataset`: path to dataset/dir <br>
  `-c`, `--classes`: path to classes.txt <br>
  `-m`, `--model`: path to ONNX model <br>
  `-s`, `--size`: Size of image used to train the model <br>
  `-conf`, `--confidence`: Model detection Confidence (0<confidence<1) <br>
  `-r`, `--remove`: List of classes need to remove

</details>

### Example:
**To .xml**
```
python3 autoAnnot.py --xml --dataset images/ --classes classes.txt --model models/model.onnx --size 224 --confidence 0.75
```
**To .txt**
```
python3 autoAnnot.py --txt --dataset images/ --classes classes.txt --model models/model.onnx --size 224 --confidence 0.75
```
**To Remove classes from auto-annotation**
```
python3 autoAnnot.py --txt --dataset images/ --classes classes.txt --model models/model.onnx --size 224 --confidence 0.75 --remove 'person' 'car
```
## 2. YOLO Model

<details>
  <summary>Args</summary>

  `-i`, `--dataset`: path to dataset/dir <br>
  `-mt`, `--model_type`: Choose YOLO Model "YOLOv7 or YOLOv8" <br>
  `-m`, `--model`: path to best.pt (YOLO) model <br>
  `-conf`, `--confidence`: Model detection Confidence (0<confidence<1) <br>
  `-r`, `--remove`: List of classes need to remove <br>
  
  for **YOLO-NAS** Model <br>
  `-t`, `--type`: Choose YOLO-NAS model type <br> **example**: `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l` <br>
  `-y`, `--yaml`: path to data.yaml file <br>
  Example: [data.yaml](https://github.com/naseemap47/YOLO-NAS/blob/master/data.yaml)
    

</details>

### Example:
**YOLOv7 Model**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov7 --model runs/train/weights/best.pt --confidence 0.8
```
- **To Remove classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov7 --model runs/train/weights/best.pt --confidence 0.8 --remove 'bus'
```

**YOLOv8 Model**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov8 --model runs/train/weights/best.pt --confidence 0.8
```
- **To Remove classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov8 --model runs/train/weights/best.pt --confidence 0.8 --remove 'elephant' 'cat' 'bear'
```

**YOLO-NAS Model**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolonas --model runs/train/weights/best.pt --type yolo_nas_s --yaml my_data/data/yaml --confidence 0.8
```
- **To Remove classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolonas --model runs/train/weights/best.pt --type yolo_nas_s --yaml my_data/data/yaml --confidence 0.8 --remove 'car'
```


## 🛠️ Tools
### 1. partition_dataset.py:
Partition your Dataset (**XML & TXT & Images**) into Train and Test in ratio

<details>
  <summary>Args</summary>
  
  `-x`, `--xml` : To partition **XML** files <br>
  `-t`, `--txt` : To partition **TXT** files <br>
  `-i`, `--imageDir` : path to image Dir, it should contain both images and Annotation files (**XML or TXT**) <br>
  `-o`, `--outputDir` : path to save Train and Test Dir, If not given - it will save inside image Dir <br>
  `-r`, `--ratio` : Ratio to partition Dataset into Train and Test (**0 < ratio < 1**)

</details>

**Example Image Dir:**
```
├── path_to/images
│   ├── 1.jpg
│   ├── 1.xml
│   ├── 2.jpeg
│   ├── 2.xml
│   ├── ...
.   .
.   .
```
```
├── path_to/images
│   ├── 1.jpeg
│   ├── 1.txt
│   ├── 2.jpg
│   ├── 2.txt
│   ├── ...
.   .
.   .
```
**Example:**
```
# XML
python3 tools/partition_dataset.py -x -i path_to/images -r 0.1

# TXT
python3 tools/partition_dataset.py -t -i path_to/images -r 0.1
```
  
### 2. txt_to_xml.py:
Convert your **TXT** Annotation files into **XML** Format

<details>
  <summary>Args</summary>
  
  `-i`, `--image` : path to image/dir <br>
  `-t`, `--txt` : path to txt/dir <br>
  `-c`, `--classes` : path to classes.txt
  
</details>
    
**classes.txt Example:**
```
car
person
apple
....
```
**Example:**
```
python3 tools/txt_to_xml.py -i path_to/imageDir -t path_to/txt_Dir -c path_to/classes.txt
```
  
### 3. xml_to_txt.py:
Convert your **XML** Annotation files into **TXT** Format

<details>
  <summary>Args</summary>
  
  `-i`, `--image` : path to image/dir <br>
  `-x`, `--xml` : path to xml/dir <br>
  `-c`, `--classes` : path to classes.txt

</details>

**classes.txt Example:**
```
car
person
apple
....
```
**Example:**
```
python3 tools/xml_to_txt.py -i path_to/imageDir -x path_to/xml_Dir -c path_to/classes.txt
```
  
### 4. xml_neg_annotation.py:
Annoatate your **Negative Dataset**

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset` : path to negative dataset <br>
  `-o`, `--save` : path to save Dir, if not exist it will create
  
</details>

**Example:**
```
python3 tools/xml_neg_annotation.py -i path_to/negDir -o path_to/saveDir
```
  
### 5. find_oneClass_from_xml.py:
To filter your **PASCAL VOC** annotaion **XML** file based on **class name**

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset` : path to negative dataset <br>
  `-o`, `--save` : path to save Dir <br>
  `-n`, `--name` : name of class, that wants filter

</details>

**Example:**
```
python3 tools/find_oneClass_from_xml.py -i path_to/dataset -o path_to/saveDir -n 'class_name'
```
  
### 6. vis_xml.py:
to visulise **xml (Pascal VOC)** annotation format

<details>
  <summary>Args</summary>
  
  `-i`, `--img` : path to image file <br>
  `-x`, `--xml` : path to xml file <br>
  `-c`, `--classes` : path to classes.txt <br>
  `--save` : to save annotated image

</details>

**Example:**
```
python3 tools/vis_xml.py -i path_to/image -x path_to/xml -c path_to_classes.txt

# to save image
python3 tools/vis_xml.py -i path_to/image -x path_to/xml -c path_to_classes.txt --save
```

### 7. vis_txt.py:
to visulise **txt (YOLO)** annotation format

<details>
  <summary>Args</summary>
  
  `-i`, `--img` : path to image file <br>
  `-t`, `--txt` : path to txt file <br>
  `-c`, `--classes` : path to classes.txt <br>
  `--save` : to save annotated image

</details>

**Example:**
```
python3 tools/vis_txt.py -i path_to/image -t path_to/txt -c path_to_classes.txt

# to save image
python3 tools/vis_txt.py -i path_to/image -t path_to/txt -c path_to_classes.txt --save
```