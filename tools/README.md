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
python3 tools/yolo_to_json.py -i path_to/imageDir -t path_to/txt_Dir -c path_to/classes.txt
```