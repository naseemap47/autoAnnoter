import cv2
import onnxruntime
import numpy as np


width = 640
height = 480

def findBBox(onnx_model_path, img, img_resize, threshold):
  
    # Saved ONNX model
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # Image
    h, w, c = img.shape
    img_resized = cv2.resize(img ,(img_resize, img_resize))
    img_data = np.reshape(img_resized, (1, img_resize, img_resize, 3))
    img_data = img_data.astype('uint8')
    ort_inputs = {input_name: img_data}
    ort_outs = session.run(None, ort_inputs)

    bbox_list = []
    class_list = []
    confidence = []
    c = 0
    for i in ort_outs[4][0]:
        if i > threshold:
            bbox = ort_outs[1][0][c]
            ymin = (bbox[0])
            xmin = (bbox[1])
            ymax = (bbox[2])
            xmax = (bbox[3])
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            xmin, ymin, xmax, ymax = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)
            bbox_list.append([xmin, ymin, xmax, ymax])

            # Detection Classes
            class_list.append(ort_outs[2][0][c])

            # confidence
            confidence.append(i)

        c = c + 1
    return bbox_list, class_list, confidence


def findClass(class_id):
  if class_id == 1:
    class_name = 'pothole'
  else:
    class_name = 'patch'
  return class_name


img = cv2.imread('img/WhatsApp Image 2022-06-16 at 8.05.18 PM (1).jpeg')
bbox_list, class_list, confidence = findBBox('model_7.onnx', img, 320, 0.4)


for bbox, id in zip(bbox_list, class_list):
    cv2.rectangle(
        img, (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        (0, 255, 0), 3
    )
    cv2.putText(
        img, f'{findClass(id)}',
        (bbox[0], bbox[1]),
        cv2.FONT_HERSHEY_PLAIN, 3,
        (0, 255, 255), 3
    )


img = cv2.resize(img, (width, height))

cv2.imshow('Img', img)
if cv2.waitKey(0) & 0xFF==ord('q'):
    cv2.destroyAllWindows()


