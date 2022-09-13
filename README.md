# autoAnnoter
autoAnnoter its a tool to auto annotate data using a exisiting model<br>
To RUN **AutoAnnotor** you can use:
 - **Command Line**
 - **Streamlit** Dashboard

## Clone this GitHub Repository
```
git clone https://github.com/naseemap47/autoAnnoter.git
cd autoAnnoter
pip3 install -r requirements.txt
```

## 1. Streamlit Dashboard
```
streamlit run app.py
```
## 2. Command Line
### For XML
```
python3 autoAnnot.py -x -i <PATH_TO_DATA> -c <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -conf <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
### For TXT
```
python3 autoAnnot.py -t -i <PATH_TO_DATA> -c <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -conf <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
### For YOLOv7
```
python3 autoAnotYolov7.py -i <PATH_TO_DATA> -m <YOLOv7_MODEL_PATH> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
