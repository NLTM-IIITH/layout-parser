import json
import os
import numpy as np
import pandas as pd

from .pinp_utils import is_box_inside, calculate_overlap_percentage

# def filter_layouts(component_after_pinp_not_ordered, layout_json_path, image_filename, width_p, header_p, footer_p):
#     image_filename = image_filename.split('/')[-1].split('.')[0]
#     with open(layout_json_path) as f:
#         layout_json = json.load(f)

#     layout_json_img = layout_json[image_filename]
        
#     #go through each component and check if it is inside Figure or Table. If inside, remove it
#     for i, row in component_after_pinp_not_ordered.iterrows():
#         tl1 = [row['Left'][0], row['Top'][1]]
#         br1 = [row['Right'][0], row['Bottom'][1]]
#         tlbr = [tl1[0], tl1[1], br1[0], br1[1]]
#         for key, values in layout_json_img.items():
#             if key == 'figure' or key == 'table' or key == 'caption' or key == 'formula':
#                 for value in values:
#                     if is_box_inside(value, tlbr, threshold_percentage=50):
#                         component_after_pinp_not_ordered.drop(i, inplace=True)
#                         # component_after_pinp_not_ordered = component_after_pinp_not_ordered.reset_index(drop=True)
#                         break
#     component_after_pinp_not_ordered = component_after_pinp_not_ordered.reset_index(drop=True)
#     print("component_after_pinp_not_ordered")
#     print(component_after_pinp_not_ordered)
#     return component_after_pinp_not_ordered

from ultralytics import YOLO
import cv2

def get_layout_from_yolo(image_filename, yolo_file_path):
    print('Running yolo inference..')
    model = YOLO(yolo_file_path)

    # Inference
    source = cv2.imread(image_filename)
    results = model(source)

    names = results[0].names
    for r in results:
        temp_dict={}
        classes = r.boxes.cls
        xyxy = r.boxes.xyxy
        # print(classes)
        print(xyxy)
        for i in range(len(classes)):
            # print(int(classes[i]))
            if names[int(classes[i])] in temp_dict:
                temp_dict[names[int(classes[i])]].append(xyxy[i].tolist())
            else:
                temp_dict[names[int(classes[i])]] = [xyxy[i].tolist()]
        print(temp_dict)
    for key, value in temp_dict.items():
        print(key, len(value))
        for v in value:
            cv2.rectangle(source, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 255, 0), 2)
            cv2.putText(source, key, (int(v[0]), int(v[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 8)
    cv2.imwrite('/data3/sreevatsa/layout-parser/saved_images/output.jpg', source)

    return temp_dict


    

def filter_layouts(component_after_pinp_not_ordered, layout_json_path, image_filename, width_p, header_p, footer_p):
    image_filename = image_filename.split('/')[-1].split('.')[0]
    with open(layout_json_path) as f:
        layout_json = json.load(f)

    layout_json_img = layout_json[image_filename]
        
    indices_to_drop = []

    # Go through each component and check if it is inside Figure or Table. If inside, mark it for removal
    for i, row in component_after_pinp_not_ordered.iterrows():
        tl1 = [row['Left'][0], row['Top'][1]]
        br1 = [row['Right'][0], row['Bottom'][1]]
        tlbr = [tl1[0], tl1[1], br1[0], br1[1]]
        for key, values in layout_json_img.items():
            if key in ['figure', 'table', 'caption', 'formula','advertisement']:
                for value in values:
                    if is_box_inside(value, tlbr, threshold_percentage=50):
                        indices_to_drop.append(i)
                        break

    # Drop the marked indices outside the loop
    component_after_pinp_not_ordered.drop(indices_to_drop, inplace=True)
    component_after_pinp_not_ordered.reset_index(drop=True, inplace=True)
    
    print("component_after_pinp_not_ordered")
    print(component_after_pinp_not_ordered)
    
    return component_after_pinp_not_ordered

#filter_layouts_direct does not use the precomputed json file with yolo inference but rather uses the yolo model to get the layout directly
def filter_layouts_direct(component_after_pinp_not_ordered, layout_json_path, image_filename, width_p, header_p, footer_p):
    image_filename = image_filename.split('/')[-1].split('.')[0]
    # with open(layout_json_path) as f:
    #     layout_json = json.load(f)

    # layout_json_img = layout_json[image_filename]
    layout_json_img = layout_json_path
        
    indices_to_drop = []

    # Go through each component and check if it is inside Figure or Table. If inside, mark it for removal
    for i, row in component_after_pinp_not_ordered.iterrows():
        tl1 = [row['Left'][0], row['Top'][1]]
        br1 = [row['Right'][0], row['Bottom'][1]]
        tlbr = [tl1[0], tl1[1], br1[0], br1[1]]
        for key, values in layout_json_img.items():
            if key in ['figure', 'table', 'caption', 'formula','advertisement']:
                for value in values:
                    if is_box_inside(value, tlbr, threshold_percentage=50):
                        indices_to_drop.append(i)
                        break

    # Drop the marked indices outside the loop
    component_after_pinp_not_ordered.drop(indices_to_drop, inplace=True)
    component_after_pinp_not_ordered.reset_index(drop=True, inplace=True)
    
    print("component_after_pinp_not_ordered")
    print(component_after_pinp_not_ordered)
    
    return component_after_pinp_not_ordered


def filter_words_layout(euclidean, layout_json_path, image_filename):
    image_filename = image_filename.split('/')[-1].split('.')[0]
    with open(layout_json_path) as f:
        layout_json = json.load(f)

    layout_json_img = layout_json[image_filename]
    
    drop_indices = []
    #go through each word and check if it is inside Figure or Table. If inside, remove it
    for i, row in euclidean.iterrows():
        print(i)
        tl1 = [row['Left'][0], row['Top'][1]]
        br1 = [row['Right'][0], row['Bottom'][1]]
        tlbr = [tl1[0], tl1[1], br1[0], br1[1]]
        for key, values in layout_json_img.items():
            if key == 'figure' or key == 'table' or key == 'caption' or key == 'formula':
                for value in values:
                    # print('value, tlbr',value, tlbr)
                    if is_box_inside(value, tlbr, threshold_percentage=50):
                        print('inside')
                        drop_indices.append(i)
                        # euclidean.drop(i, inplace=True)
                        # euclidean = euclidean.reset_index(drop=True)
                        break
    euclidean.drop(drop_indices, inplace=True)
    # euclidean.reset_index(drop=True, inplace=True)
    return euclidean
