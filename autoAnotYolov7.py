import cv2
from utils.hubconf import custom


# Load YOLOv7 Model (best.pt)
model = custom(path_or_model='best.pt')  # custom example

img = cv2.imread('/home/naseem/My-Project/toolbuilder/autoAnnoter/img/WhatsApp Image 2022-06-16 at 8.05.04 PM.jpeg')
results = model(img)

# Bounding Box
box = results.pandas().xyxy[0]
x = box['xmin']
y = box['ymin']
w = box['xmax'] - x  # x+w
h = box['ymax'] - y  # y+h
confidence = box['confidence']
Class = box['class']
name = box['name']

# Bounding Box Parameters
para = []
for i in box.index:
    x, y, w, h, c, n = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i] - box['xmin'][i]), \
                        int(box['ymax'][i] - box['ymin'][i]), box['confidence'][i], (box['name'][i])
    para.append([x, y, w, h, c, n])

# print(para)

for p in para:
    cv2.rectangle(
        img, (x, y), ((x+w), (y+h)),
        (0, 255, 255), 3
    )

cv2.imshow('Image', img)
if cv2.waitKey(0) & 0xFF==ord('q'):
    cv2.destroyAllWindows()

