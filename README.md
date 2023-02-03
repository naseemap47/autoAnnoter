# autoAnnoter
autoAnnoter its a tool to auto annotate data using a exisiting model

## 🚀 New Update (03-02-2023)
- ### **YOLOv8** Auto Annotation
  - Auto Annotate using YOLOv8 Model 
## Updates
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
  `-conf`, `--confidence`: Model detection Confidence (0<confidence<1)

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
## 2. YOLO Model

<details>
  <summary>Args</summary>

  `-i`, `--dataset`: path to dataset/dir <br>
  `-mt`, `--model_type`: Choose YOLO Model "YOLOv7 or YOLOv8" <br>
  `-m`, `--model`: path to best.pt (YOLO) model <br>
  `-conf`, `--confidence`: Model detection Confidence (0<confidence<1)

</details>

### Example:
**YOLOv7 Model**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov7 --model runs/train/weights/best.pt --confidence 0.8
```

**YOLOv8 Model**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov8 --model runs/train/weights/best.pt --confidence 0.8
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
  
  `-i`, `--image` : path to image/dir
  `-x`, `--xml` : path to xml/dir
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
  
  `-i`, `--dataset` : path to negative dataset
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
  
  `-i`, `--dataset` : path to negative dataset
  `-o`, `--save` : path to save Dir
  `-n`, `--name` : name of class, that wants filter

</details>

**Example:**
```
python3 tools/find_oneClass_from_xml.py -i path_to/dataset -o path_to/saveDir -n 'class_name'
```
  
