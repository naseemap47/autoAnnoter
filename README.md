# autoAnnoter
[<img src="https://img.shields.io/badge/Docker-Image-blue.svg?logo=docker">](<https://hub.docker.com/repository/docker/naseemap47/auto-annoter>)<br>

autoAnnoter: Its a tool to auto annotate data using a exisiting model

## Auto annotate any class 🚀 New Update (17-08-2023)
### 🥳 Auto Annotate using OWL-ViT Model
We can auto annotate **any class** on our data using new OWL-ViT model.<br>
No need of any pre-trained or any custom model to auto-annotate.

### About OWL-ViT
**OWL-ViT** is an open-vocabulary object detector. It means that it can detect objects in images based on free-text queries without the need to fine-tune the model on labeled datasets.

## Updates
<details>
  <summary>Show</summary>
  
- (**12-06-2023**) **Grounding DINO** 🦕 auto annotate any class
- (**01-06-2023**) **YOLO-NAS** Auto Annotation
  - Auto Annotate using YOLO-NAS Model
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

</details>

## Clone this GitHub Repository
```
git clone https://github.com/naseemap47/autoAnnoter.git
```
## Install Dependencies
**Recommended**:
```
conda create -n auto python=3.9 -y
conda activate auto
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```
#### OR
```
cd autoAnnoter/
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## Zero-shot object detection
### 1. Grounding DINO 🦕
**Grounding DINO** is text to detection model. So we need to give text prompt that correspond to respective class.<br>
### 2. OWL-ViT (Recommended)
**OWL-ViT** is an open-vocabulary object detector. It means that it can detect objects in images based on free-text queries without the need to fine-tune the model on labeled datasets..<br>

## Auto annotate any class
To do this, we needs to create [prompt.json](https://github.com/naseemap47/autoAnnoter/blob/dino/prompt.json) <br>

**JSON keys** should be text prompt to Grounding DINO model.<br>
But the values for the each keys should be **class names** for that detection.<br>

### Example:
Here I need to train one custom model that can predict **high quality cap** and **low quality cap**.<br>
So for this I give my **Grounding DINO/OWL-ViT** text prompt as **red cap** and **yellow caps**, to annotate my **high quality cap** and **low quality cap** classes.<br>
I give this example to show you that, some times we need to give **Grounding DINO/OWL-ViT** text prompt as more elaborate way, like my example.

![out11](https://github.com/naseemap47/autoAnnoter/assets/88816150/df2ebb71-ad67-4bce-9099-cd9857a4cfcd)


```
{
    "red caps": "high quality cap",
    "yellow caps": "low quality cap"
}
``` 
### This approch we can use when the object is same, but have different feature like color.
### 1. OWL-ViT (Recommended)

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset`: path to dataset/dir <br>
  `-p`, `--prompt`: path to prompt.json <br>
  `-bt`, `--box_thld`: bounding box Threshold
  
</details>

To auto-annotate **OWL-ViT** model, we need to give text prompt that correspond to respective class. 

**Example:**
```
python3 owlvit.py --dataset images/ --prompt prompt.json
```

### 2. Grounding DINO 🦕

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset`: path to dataset/dir <br>
  `-p`, `--prompt`: path to prompt.json <br>
  `-bt`, `--box_thld`: Box Threshold <br>
  `-tt`, `--txt_thld`: text threshold
  
</details>

To auto-annotate Grounding DINO model, we need to give text prompt that correspond to respective class. 

**Example:**
```
python3 dino.py --dataset images/ --prompt prompt.json
```

## 2. ONNX Model

<details>
  <summary>Args</summary>
  
  `-x`, `--xml`: to annotate in XML format <br>
  `-t`, `--txt`: to annotate in (.txt) format <br>
  `-i`, `--dataset`: path to dataset/dir <br>
  `-c`, `--classes`: path to classes.txt <br>
  `-m`, `--model`: path to ONNX model <br>
  `-s`, `--size`: Size of image used to train the model <br>
  `-conf`, `--confidence`: Model detection Confidence (0<confidence<1) <br>
  `-r`, `--remove`: List of classes need to remove <br>
  `-k`, `--keep`: List of classes need to keep

</details>

### Example:
**To .xml**
```
python3 autoAnnot.py --xml --dataset images/ --classes classes.txt \
                     --model models/model.onnx --size 224 --confidence 0.75
```
**To .txt**
```
python3 autoAnnot.py --txt --dataset images/ --classes classes.txt \
                     --model models/model.onnx --size 224 --confidence 0.75
```
**To Remove classes from auto-annotation**
```
python3 autoAnnot.py --txt --dataset images/ --classes classes.txt \
                     --model models/model.onnx --size 224 --confidence 0.75 \
                     --remove 'person' 'car
```
**To Keep classes from auto-annotation**
```
python3 autoAnnot.py --txt --dataset images/ --classes classes.txt \
                     --model models/model.onnx --size 224 --confidence 0.75 \
                     --keep 'person' 'car
```
## 3. YOLO Model

<details>
  <summary>Args</summary>

  `-i`, `--dataset`: path to dataset/dir <br>
  `-mt`, `--model_type`: Choose YOLO Model "YOLOv7 or YOLOv8" <br>
  `-m`, `--model`: path to best.pt (YOLO) model <br>
  `-conf`, `--confidence`: Model detection Confidence (0<confidence<1) <br>
  `-r`, `--remove`: List of classes need to remove <br>
  `-k`, `--keep`: List of classes need to keep <br>
  
  for **YOLO-NAS** Model <br>
  `-t`, `--type`: Choose YOLO-NAS model type <br> **example**: `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l` <br>
  `-n`, `--num`: number of classes that model trained on
    

</details>

## Examples:
### 1. YOLOv7 Model
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov7 \
                        --model runs/train/weights/best.pt --confidence 0.8
```
- **To Remove classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov7 \
                        --model runs/train/weights/best.pt --confidence 0.8 \
                        --remove 'bus'
```
- **To Keep classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov7 \
                        --model runs/train/weights/best.pt --confidence 0.8 \
                        --keep 'bus'
```

### 2. YOLOv8 Model
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov8 \
                        --model runs/train/weights/best.pt --confidence 0.8
```
- **To Remove classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov8 \
                        --model runs/train/weights/best.pt --confidence 0.8 \
                        --remove 'elephant' 'cat' 'bear'
```
- **To Keep classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolov8 \
                        --model runs/train/weights/best.pt --confidence 0.8 \
                        --keep 'cat'
```

### 3. YOLO-NAS Model
#### Custom Model
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolonas \
                        --model runs/train/weights/best.pt --type yolo_nas_s \
                        --num 8 --confidence 0.8
```
#### COCO Model
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolonas \
                        --model coco --type yolo_nas_s \
                        --confidence 0.8
```

- **To Remove classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolonas \
                        --model runs/train/weights/best.pt --type yolo_nas_s \
                        --num 80 --confidence 0.8 \
                        --remove 'car'
```
- **To Keep classes from auto-annotation**
```
python3 autoAnotYolo.py --dataset dataset/images --model_type yolonas \
                        --model runs/train/weights/best.pt --type yolo_nas_s \
                        --num 32 --confidence 0.8 \
                        --keep 'car'
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

### 8. create_yolo.py:
to convert mixed images & labels into respective directory of images and labels

<details>
  <summary>Args</summary>
  
  `-i`, `--data` : path to image/dir/data

</details>

**Example:**
```
python3 tools/create_yolo.py -i path_to/data Dir
```

### 9. yolo_to_kitti.py:
to convert YOLO annotations (.txt) to KITTI format

<details>
  <summary>Args</summary>
  
  `-i`, `--img` : path to image file <br>
  `-t`, `--txt` : path to txt file <br>
  `-c`, `--classes` : path to classes.txt

</details>

**Example:**
```
python3 tools/yolo_to_kitti.py -i path_to/image -t path_to/txt -c path_to_classes.txt
```

### 10. yolo_to_json.py:
Convert your **TXT (YOLO)** Annotation files into **COCO JSON** Format

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
python3 tools/txt_to_json.py -i path_to/imageDir -t path_to/txt_Dir -c path_to/classes.txt
```
