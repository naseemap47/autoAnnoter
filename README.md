# autoAnnoter
[<img src="https://img.shields.io/badge/Docker-Image-blue.svg?logo=docker">](<https://hub.docker.com/repository/docker/naseemap47/auto-annoter>)<br>

autoAnnoter: Its a tool to auto annotate data using a exisiting model

### 🛠️ Tools
- [partition_dataset.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#1-partition_datasetpy)
- [txt_to_xml.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#2-txt_to_xmlpy)
- [xml_to_txt.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#3-xml_to_txtpy)
- [xml_neg_annotation.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#4-xml_neg_annotationpy)
- [find_oneClass_from_xml.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#5-find_oneclass_from_xmlpy)
- [vis_xml.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#6-vis_xmlpy)
- [vis_txt.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#7-vis_txtpy)
- [create_yolo.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#8-create_yolopy)
- [yolo_to_kitti.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#9-yolo_to_kittipy)
- [yolo_to_json.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#10-yolo_to_jsonpy)
- [split_data.py](https://github.com/naseemap47/autoAnnoter/tree/aug/tools#11-split_datapy)

### 🛠️ Augumentaion
Augument your annotation files (Object detection) PASCAL VOC (XML) or YOLO (TXT)
- [XML Augmentation](https://github.com/naseemap47/autoAnnoter/tree/aug/Augmentation#xml-augmentation)
- [YOLO Augmentation (TXT)](https://github.com/naseemap47/autoAnnoter/tree/aug/Augmentation#yolo-augmentation-txt)

## Auto annotate any class 🚀
### Open-Vocabulary Detection Model
- Grounding DINO
- OWL-ViT
- YOLO-World
- PaliGemma
- Florence-2

## Open-Vocabulary Detection
### 1. Grounding DINO 🦕
**Grounding DINO** is text to detection model. So we need to give text prompt that correspond to respective class.<br>
### 2. OWL-ViT
**OWL-ViT** is an open-vocabulary object detector. It means that it can detect objects in images based on free-text queries without the need to fine-tune the model on labeled datasets..<br>
### 3. YOLO-World
The **YOLO-World** Model introduces an advanced, real-time Ultralytics YOLOv8-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts.


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

## Auto annotate any class with Open-Vocabulary Detection Model
To do this, we needs to create [prompt.json](https://github.com/naseemap47/autoAnnoter/blob/dino/prompt.json) <br>

**JSON keys** should be text prompt to Open-Vocabulary Detection Model.<br>
But the values for the each keys should be **class names** for that detection.<br>

### Example:
Here I need to train one custom model that can predict **high quality cap** and **low quality cap**.<br>
So for this I give my **Open-Vocabulary Detection Model** text prompt as **red cap** and **yellow caps**, to annotate my **high quality cap** and **low quality cap** classes.<br>

I give this example to show you that, some times we need to give **Open-Vocabulary Detection Model** text prompt as more elaborate way, like my example.

![out11](https://github.com/naseemap47/autoAnnoter/assets/88816150/df2ebb71-ad67-4bce-9099-cd9857a4cfcd)


```
{
    "red caps": "high quality cap",
    "yellow caps": "low quality cap"
}
``` 
### This approch we can use when the object is same, but have different feature like color.

### 1. Grounding DINO 🦕

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

### 2. OWL-ViT

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

### 3. YOLO-World

<details>
  <summary>Args</summary>
  
  `-i`, `--data`: path to dataset/dir <br>
  `-p`, `--prompt`: path to prompt.json <br>
  `-c`, `--conf`: detection confidence <br>
  `-m`, `--model`: Choose model version/type

    'yolov8s-world.pt', 'yolov8s-worldv2.pt'
    'yolov8m-world.pt', 'yolov8m-worldv2.pt'
    'yolov8l-world.pt', 'yolov8l-worldv2.pt'
    'yolov8x-world.pt', 'yolov8x-worldv2.pt
  `-f`, `--format`: annotation format
  
    'txt', 'xml'
  
</details>

To auto-annotate **YOLO-World** model, we need to give text prompt that correspond to respective class. 

**Example:**
```
python3 owlvit.py --data images/ --prompt prompt.json --conf 0.8 \
                  --model yolov8m-worldv2.pt --format txt
```

## Auto annotate any class with Pre-Trained Detection Model
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
## 2. YOLO Model

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
