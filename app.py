import streamlit as st
from anot_utils import findBBox, save_xml, save_yolo, get_BBoxYOLOv7
from utils.hubconf import custom
import cv2
import onnxruntime
import glob
import os


st.title('Auto Annotator')
st.sidebar.title('Settings')
options = st.sidebar.radio(
    'Model Options:',
    ('ONNX', 'YOLOv7')
)

if options=='ONNX':
    out_options = st.sidebar.radio(
    'Annotation Format:',
    ('XML', 'TXT')
    )
    # path to ONNX model
    onnx_model_path = st.text_input(
        'path to Model:',
        'eg: dir/model.onnx'
    )
    load_model = st.checkbox('Load ONNX Model')

    path_to_dir = st.text_input(
        'Path to Dataset'
    )

    img_size = st.number_input(
        'Size of image used to train the model'
    )

    detect_conf = st.slider(
        'Model detection Confidence',
        min_value=0.01, max_value=0.99,
        value=0.4
    )

    # ONNX Model
    if load_model:
        onnx_session = onnxruntime.InferenceSession(onnx_model_path)
        st.success('ONNX Model Loaded Successfully')

        img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
            glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
            glob.glob(os.path.join(path_to_dir, '*.png'))

        # XML Annotation
        if out_options=="XML":
            # Class txt
            path_to_txt = st.file_uploader(
                'Class.txt:', type=['txt']
            )
            if path_to_txt is not None:
                for img in img_list:
                    image = cv2.imread(img)
                    h, w, c = image.shape
                    bbox_list, class_list, confidence = findBBox(
                        onnx_session, image, int(img_size), detect_conf)
                    folder_name, file_name = os.path.split(img)

                    bytes_data = path_to_txt.getvalue()
                    class_names = bytes_data.decode('utf-8').split("\n")

                    save_xml(folder_name, file_name, img, w, h, c,
                            bbox_list, class_list, class_names)

                    st.success(f'Successfully Annotated {file_name}')

                st.success('XML-Auto_Annotation Successfully Completed')

        if out_options=="TXT":
            if st.checkbox('RUN'):
                for img in img_list:
                    image = cv2.imread(img)
                    h, w, c = image.shape
                    bbox_list, class_list, confidence = findBBox(
                        onnx_session, image, int(img_size), detect_conf)
                    folder_name, file_name = os.path.split(img)
                    save_yolo(folder_name, file_name, w, h, bbox_list, class_list)
                    st.success(f'Successfully Annotated {file_name}')

                st.success('TXT-Auto_Annotation Successfully Completed')

if options=='YOLOv7':
    out_options = st.sidebar.radio('Annotation Format:', ['TXT'])
    
    # path to YOLOv7 model
    yolov7_model_path = st.text_input(
        'path to YOLOv7 Model:',
        'eg: dir/yolov7.pt'
    )
    load_model = st.checkbox('Load YOLOv7 Model')

    path_to_dir = st.text_input(
        'Path to Dataset'
    )

    detect_conf = st.slider(
        'Model detection Confidence',
        min_value=0.01, max_value=0.99,
        value=0.4
    )

    # Load YOLOv7 Model (best.pt)
    if load_model:
        model = custom(path_or_model=yolov7_model_path)
        st.success('YOLOv7 Model Loaded Successfully')

        img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
            glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
            glob.glob(os.path.join(path_to_dir, '*.png'))

        if st.checkbox('RUN'):
            for img in img_list:
                folder_name, file_name = os.path.split(img)
                image = cv2.imread(img)
                h, w, c = image.shape
                bbox_list, class_list, confidence = get_BBoxYOLOv7(image, model, detect_conf)
                save_yolo(folder_name, file_name, w, h, bbox_list, class_list)

                st.success(f'Successfully Annotated {file_name}')

            st.success('YOLOv7-Auto_Annotation Successfully Completed')
