import os


def save_yolo(folder_name, file_name, w, h, bbox_list, class_list):
    txt_name = os.path.splitext(file_name)[0] + '.txt'
    path_to_save = os.path.join(folder_name, txt_name)
    out_file = open(path_to_save, 'w')
    for box, class_index in zip(bbox_list, class_list):
        x_min = box[0]
        x_max = box[1]
        y_min = box[2]
        y_max = box[3]

        x_center = float((x_min + x_max)) / 2 / w
        y_center = float((y_min + y_max)) / 2 / h

        width = float((x_max - x_min)) / w
        height = float((y_max - y_min)) / h

        # Save
        out_file.write("%d %.6f %.6f %.6f %.6f\n" % (int(class_index)-1, x_center, y_center, width, height))

    print(f'Successfully Created {txt_name}')

