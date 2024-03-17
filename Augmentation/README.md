# 🛠️ Augumentaion

Augument your annotation files (Object detection) PASCAL VOC (XML) or YOLO (TXT)

- HorizontalFlip
- RandomScale
- Scale
- RandomTranslate
- Translate
- RandomRotate
- Rotate
- RandomShear
- Shear
- Resize
- RandomHSV
- Sequence


## XML Augmentation
### 1. xml_aug.py:
**Arguments:**
  
  `-i`, `--image` : path to image/dir <br>
  `-x`, `--xml` : path to xml/dir <br>
  `-s`, `--save` : path to save XML Augmentation <br>
  `-y`, `--yaml` : path to aug yaml file

**Example:**
```sh
python3 Augmentation/xml_aug.py -i path_to/images -x path_to/xml_labels -s path_to_save -y path_to_aug_yaml

# Example
python3 Augmentation/xml_aug.py -i Augmentation/sample/images -x Augmentation/sample/labels -s Augmentation/sample/Aug -y Augmentation/default.yaml
```

## YOLO Augmentation (TXT)
### 2. txt_aug.py:
**Arguments:**
  
  `-i`, `--image` : path to image/dir <br>
  `-t`, `--txt` : path to txt/dir <br>
  `-s`, `--save` : path to save YOLO(txt) Augmentation <br>
  `-y`, `--yaml` : path to aug yaml file

**Example:**
```sh
python3 Augmentation/txt_aug.py -i path_to/images -t path_to/txt_labels -s path_to_save -y path_to_aug_yaml

# Example
python3 Augmentation/txt_aug.py -i Augmentation/sample/images -t Augmentation/sample/labels -s Augmentation/sample/Aug -y Augmentation/default.yaml
```

## Augmentation YAML
sample Augmentation YAML file: [default.yaml](https://github.com/naseemap47/autoAnnoter/blob/aug/Augmentation/default.yaml)
```yaml
# ----------------- Augmentation Parameters -----------------
# image HSV-Hue augmentation
hsv_h: 
  hue: 100          # Range (0-179)
  prob: 0.4         # image HSV-Hue augmentation (probability)

# image HSV-Saturation augmentation
hsv_s:
  saturation: 212   # Range (0-255)
  prob: 0.7         # image HSV-Saturation augmentation (probability)

# image HSV-Value (brightness) augmentation
hsv_v:
  brightness: 120   # Range (0-255)
  prob: 0.4         # image HSV-Value (brightness) augmentation (probability)

# Mixed image HSV augmentation (Mixed HSV-Hue, HSV-Saturation and HSV-Value (brightness))
hsv:
  hue: 2          # Range (0-179)
  saturation: 2   # Range (0-255)
  brightness: 2   # Range (0-255)
  prob: 0.4       # Mixed image HSV augmentation (probability)

# image rotation
degrees:
  deg: 10         # image rotation (+/- deg)
  prob: 0.2       # image rotation (probability)

# Random image rotation
degrees_random:
  deg: 10         # image rotation range (+deg, -deg)
  prob: 0.2       # image rotation (probability)

# (float) image translation (+/- fraction)
translate:
  translate_x: 0.2
  translate_y: 0.2
  prob: 0.1

# (float) image Random translation (+/- fraction)
translate_random:
  translate: 0.2
  prob: 0.1

# (float) image scale (+/- gain)
scale:
  scale_x: 0.2
  scale_x: 0.2
  prob: 0.5

# (float) image Random scale (+/- gain)
scale_random:
  scale: 0.2
  prob: 0.5

# (float) image shear (+/- deg)
shear:
  shear: 0.2
  prob: 0.2

# (float) Random image shear (+/- deg)
shear_random:
  shear: 0.2
  prob: 0.2

flipud: 0.0 # (float) image flip up-down (probability)
fliplr: 0.5 # (float) image flip left-right (probability)

```
