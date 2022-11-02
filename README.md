# autoAnnoter
autoAnnoter its a tool to auto annotate data using a exisiting model

## 🚀 New Update
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
### To XML Format
```
python3 autoAnnot.py -x -i <PATH_TO_DATA> -t <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
### To TXT Format
```
python3 autoAnnot.py -y -i <PATH_TO_DATA> -t <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
## 2. YOLOv7 Model
```
python3 autoAnotYolov7.py -i <PATH_TO_DATA> -m <YOLOv7_MODEL_PATH> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```

## 🛠️ Tools
- **1. partition_dataset.py** : Partition your Dataset (**XML & TXT & Images**) into Train and Test in ratio
  - `-x`, `--xml` : To partition **XML** files
  - `-t`, `--txt` : To partition **TXT** files
  - `-i`, `--imageDir` : path to image Dir, it should contain both images and Annotation files (**XML or TXT**)
  - `-o`, `--outputDir` : path to save Train and Test Dir, If not given - it will save inside image Dir
  - `-r`, `--ratio` : Ratio to partition Dataset into Train and Test (**0 < ratio < 1**)
  - **Example Image Dir:**
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
  
- **2. txt_to_xml.py** : Convert your **TXT** Annotation files into **XML** Format
  - `-i`, `--image` : path to image/dir
  - `-t`, `--txt` : path to txt/dir
  - `-c`, `--classes` : path to classes.txt
  
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
  
- **3. xml_to_txt.py** : Convert your **XML** Annotation files into **TXT** Format
  - `-i`, `--image` : path to image/dir
  - `-x`, `--xml` : path to xml/dir
  - `-c`, `--classes` : path to classes.txt
  
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
  
- **4. xml_neg_annotation.py** : Annoatate your **Negative Dataset**
  - `-i`, `--dataset` : path to negative dataset
  - `-o`, `--save` : path to save Dir, if not exist it will create
  
  **Example:**
  ```
  python3 tools/xml_neg_annotation.py -i path_to/negDir -o path_to/saveDir
  ```
  
- **5. find_oneClass_from_xml.py** : To filter your **PASCAL VOC** annotaion **XML** file based on **class name**
  - `-i`, `--dataset` : path to negative dataset
  - `-o`, `--save` : path to save Dir
  - `-n`, `--name` : name of class, that wants filter
  
  **Example:**
  ```
  python3 tools/find_oneClass_from_xml.py -i path_to/dataset -o path_to/saveDir -n 'class_name'
  ```
