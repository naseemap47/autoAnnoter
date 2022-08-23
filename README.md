# autoAnnoter
autoAnnoter its a tool to auto annotate data using a exisiting model

## Streamlit Dashboard:
https://naseemap47-autoannoter-app-streamlit-h18l87.streamlitapp.com/

## Clone this GitHub Repository
```
git clone https://github.com/naseemap47/autoAnnoter.git
```
## For XML
```
python3 autoAnnot.py -x -i <PATH_TO_DATA> -t <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
## For TXT
```
python3 autoAnnot.py -y -i <PATH_TO_DATA> -t <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
## For YOLOv7
```
python3 autoAnotYolov7.py -i <PATH_TO_DATA> -m <YOLOv7_MODEL_PATH> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
