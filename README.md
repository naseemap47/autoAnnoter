# autoAnnoter
autoAnnoter its a tool to auto annotate data using a exisiting model

## For XML
```
python3 autoAnnot.py -x -i <PATH_TO_DATA> -t <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
## For YOLO
```
python3 autoAnnot.py -y -i <PATH_TO_DATA> -t <PATH_TO_classes.txt> -m <ONNX_MODEL_PATH> -s <SIZE_OF_IMAGE_WHEN_TRAIN_YOUR_MODEL> -c <MODEL_OBJCET_DETECTION_CONFIDENCE>
```
