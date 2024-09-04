from .pinp_utils import *
from .dist_utils import *
from .kde_utils import *
from .para_utils import *
import cv2
import pandas as pd
import networkx as nx
import os

from .layout_filtering import *

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

def get_paras(G):
    df = pd.DataFrame()
    connected_components = list(nx.connected_components(G))

    connected_components_list = [list(component) for component in connected_components]

    data = {"connected_components": connected_components_list}
    return data
    # json_data = json.dumps(data)

    # file_path = "paragraph.json"

    # with open(file_path, "w") as json_file:
    #     json_file.write(json_data)


def recognise_paragraphs(image, target_components, euclidean, image_filename, width_p, header_p, footer_p, layout_json_path):

    #  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    component = pd.DataFrame()
    count = 0
    # Create a copy of the image to draw the boxes and labels on
    #  image_with_boxes = image_rgb.copy()
    for i in target_components:

        left1 = []
        right1 = []
        top1 = []
        bottom1 = []
        for index, row in euclidean.iterrows():
            box_id = int(row['Id'])
            if box_id in i[0]:
                right_box1 = row['Right']
                left_box1 = row['Left']
                top_box1 = row['Top']
                bottom_box1 = row['Bottom']
                right_box = right_box1[0]
                left_box = left_box1[0]
                top_box = top_box1[1]
                bottom_box = bottom_box1[1]
                # print(left_box1, right_box1, top_box1, bottom_box1)
                # print(type(left_box1), type(right_box1), type(top_box1), type(bottom_box1))
                # right_box = parse_string(right_box1,"[",",")
                # left_box = parse_string(left_box1,"[",",")
                # top_box = parse_string(top_box1,",","]")
                # bottom_box = parse_string(bottom_box1,",","]")
                if(int(round(float(left_box)))!=-1):
                    left1.append(int(round(float(left_box))))
                if(int(round(float(right_box)))!=-1):
                    right1.append(int(round(float(right_box))))
                if(int(round(float(top_box)))!=-1):
                    top1.append(int(round(float(top_box))))
                if(int(round(float(bottom_box)))!=-1):
                    bottom1.append(int(round(float(bottom_box))))
        l = min(left1)
        r = max(right1)
        t = min(top1)
        b = max(bottom1)
        center_top = [int(l+r)/2, int(t)]
        center_bottom = [int(l+r)/2, int(b)]
        center_right = [int(r), int(t+b)/2]
        center_left = [int(l), int(t+b)/2]
        larger_box_top_left = (int(l - 20), int(t - 20))
        larger_box_bottom_right = (int(r + 10), int(b + 10))
        bottom_box = [-1, 0]
        visited = 0
        order = -1
        # cv2.rectangle(image_with_boxes, larger_box_top_left, larger_box_bottom_right, (0, 0, 255), 2)
        new_row = pd.Series([i, count, center_top, center_bottom, center_right, center_left, bottom_box, visited, order])
        component = component._append(new_row, ignore_index=True)
        count = count+1
    #  plt.imshow(image_with_boxes)
    #  plt.axis('off')
    #  plt.show()
    #  output_path = 'Para.png'
    #  cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    new_column_names = {0: 'Component', 1: 'Id', 2: 'Top',3: 'Bottom',4: 'Right',5: 'Left',6: 'Bottom_Box',7: 'Visited',8: 'Order'}
    component = component.rename(columns=new_column_names)
    component = ignore_margins(component,width_p,header_p,footer_p,image_filename)
    component0 = component.reset_index(drop=True)

    component_before_pinp = component0.copy()

    #add visualisation for paragraph before para in para fix -> shows the paragraphs before pinp
    draw_rectangles_paras_cv2(component_before_pinp, image_filename, "before_pinp")
    # print("component before pinp")
    # print(component_before_pinp)


    '''
    Para in para FIX (post-processing) for the smaller 
    wrongly identified paragraphs that are inside the larger paragraphs
    '''
    #adding paragraph in paragraph fix here 
    if len(component0) >=3:
        component_after_pinp_not_ordered_first = pinp(component0, width_p, header_p, footer_p, image_filename, theta=7)
        # print("component after pinp but not ordered first")
        # print(component_after_pinp_not_ordered_first)
        draw_rectangles_paras_cv2(component_after_pinp_not_ordered_first, image_filename, "after_pinp_not_ordered_first")
        
        component_after_pinp_not_ordered_second = pinp2(component_after_pinp_not_ordered_first, width_p, header_p, footer_p, image_filename, theta=10)
        component_after_pinp_not_ordered = pinp(component_after_pinp_not_ordered_second, width_p, header_p, footer_p, image_filename, theta=7)

        # print("component after pinp but not ordered")
        # print(component_after_pinp_not_ordered)
        draw_rectangles_paras_cv2(component_after_pinp_not_ordered, image_filename, "after_pinp_not_ordered")
    else:
        component_after_pinp_not_ordered = component0
    # component = component0
    
    '''
    Layout Filtering
    '''
    #use layouts to filter out the paragraphs that are wrongly detected in figures and tables and are not in the main text area
    if layout_json_path is not None:
        component_after_pinp_not_ordered_after_layout = filter_layouts_direct(component_after_pinp_not_ordered, layout_json_path, image_filename, width_p, header_p, footer_p) #change to filter_layouts to use the precomputed json file to filter out the paragraphs 
        return component_before_pinp,component_after_pinp_not_ordered, component_after_pinp_not_ordered_after_layout
    elif layout_json_path is None:
        return component_before_pinp,component_after_pinp_not_ordered, None

    '''
    - component_after_pinp_not_ordered is the component after the pinp fix but not ordered
    variable name for component_after_pinp_not_ordered was component before, below commented code used to have component 
    - moved the below find_closest_paragraphs(), kde_estimate() and calculate_bottombox_para() to pinp(), so commented below code
    '''
    
    # vertical = find_closest_paragraphs(component) #added
    # print("Vertical distances between paragraphs", vertical)
    # x_para = kde_estimate(vertical) #added
    # print("KDE estimate of vertical distances between paragraphs", x_para)
    # calculate_bottombox_para(component,x_para) #added
    
    # return component
    # return component_before_pinp,component_after_pinp_not_ordered, component_after_pinp_not_ordered_after_layout

##### PARAGRAPH ORDER

# def paragraph_order(component):
#     comp = component
#     order = 0
#     min_idx = minimum_euclidean(comp)

#     while any(comp['Visited'] == 0) and min_idx != -1:
#         if comp['Visited'][min_idx] != 1:
#             # component['Visited'][min_idx] = 1
#             # component['Order'][min_idx] = order
#             comp.loc[min_idx, 'Visited'] = 1
#             comp.loc[min_idx, 'Order'] = order
#             order += 1

#         next_idx = get_next(comp, min_idx)
#         if next_idx != -1:
#             min_idx = next_idx
#         else:
#             min_idx = minimum_euclidean(comp)

#     return comp

# def paragraph_order(component):
#     comp = component
#     order = 0
#     min_idx = minimum_euclidean(comp)
#     print('before loop')
#     while any(comp['Visited'] == 0) and min_idx != -1:
#         print("ii")
#         if comp['Visited'][min_idx] != 1:
#             comp.loc[min_idx, 'Visited'] = 1
#             comp.loc[min_idx, 'Order'] = order
#             order += 1
#             print(order)
#             if order >= len(comp):
#                 break
#         print("minn")
#         next_idx = get_next(comp, min_idx)
#         if next_idx != -1:
#             min_idx = next_idx
#         else:
#             min_idx = minimum_euclidean(comp)
#         print("loop end")
#     return comp


def paragraph_order(component):
    print('in para order')
    
    comp = component
    order = 0
    min_idx = minimum_euclidean(comp)
    print('Min_idx:',min_idx)
    # print('before loop')
    print("length of component:", len(comp))
    for i in range(len(comp)):
        # print("i:", i)
        # print(comp)
        # print("MM",min_idx)
        # print(comp)
        if any(comp['Visited'] == 0) and min_idx != -1:
            # print("ii")
            if comp['Visited'][min_idx] != 1:
                comp.loc[min_idx, 'Visited'] = 1
                comp.loc[min_idx, 'Order'] = order
                order += 1
                # print(order)
            # print("minn")
            next_idx = get_next(comp, min_idx)
            if next_idx != -1:
                min_idx = next_idx
            else:
                min_idx = minimum_euclidean(comp)
            # print("loop end")
        else:
            break
            
    return comp


def paragraph_order_2(component):
    print('in para order')
    
    comp = component.copy()  # Make a copy of the component to avoid modifying the original DataFrame
    order = 0
    min_idx = minimum_euclidean(comp)
    print('Min_idx:', min_idx)
    print("Length of component:", len(comp))
    
    # Iterate until all paragraphs are visited
    while comp['Visited'].eq(0).any() and min_idx != -1:
        if comp.loc[min_idx, 'Visited'] != 1:
            comp.loc[min_idx, 'Visited'] = 1
            comp.loc[min_idx, 'Order'] = order
            order += 1
        
        next_idx = get_next(comp, min_idx)
        if next_idx != -1:
            min_idx = next_idx
        else:
            min_idx = minimum_euclidean(comp)
    
    return comp


#### VISUALISE PARAGRAPH ORDER


def visualise_paragraph_order(image_file_name, image, target_components, euclidean,component):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    count = 0
    image_with_boxes = image_rgb.copy()
    # print(type(component))
    for idx1,rowcomp in component.iterrows():
        left1 = []
        right1 = []
        top1 = []
        bottom1 = []
        for index, row in euclidean.iterrows():
            box_id = int(row['Id'])
            # print(type(box_id))
            # print(type(rowcomp['Component'][0][0]))
            if box_id in rowcomp['Component'][0]:
                order = rowcomp['Order']
                right_box1 = row['Right']
                left_box1 = row['Left']
                top_box1 = row['Top']
                bottom_box1 = row['Bottom']
                right_box = right_box1[0]
                left_box = left_box1[0]
                top_box = top_box1[1]
                bottom_box = bottom_box1[1]
                # print(right_box,left_box,top_box,bottom_box)
                # right_box = parse_string(right_box1,"[",",")
                # left_box = parse_string(left_box1,"[",",")
                # top_box = parse_string(top_box1,",","]")
                # bottom_box = parse_string(bottom_box1,",","]")
                if(int(round(float(left_box)))!=-1):
                    left1.append(int(round(float(left_box))))
                if(int(round(float(right_box)))!=-1):
                    right1.append(int(round(float(right_box))))
                if(int(round(float(top_box)))!=-1):
                    top1.append(int(round(float(top_box))))
                if(int(round(float(bottom_box)))!=-1):
                    bottom1.append(int(round(float(bottom_box))))
            # print(left1,right1,top1,bottom1)  
        l = min(left1)
        r = max(right1)
        t = min(top1)
        b = max(bottom1)
        center_top = [int(l+r)/2, int(t)]
        center_bottom = [int(l+r)/2, int(b)]
        center_right = [int(r), int(t+b)/2]
        center_left = [int(l), int(t+b)/2]
        larger_box_top_left = (int(l - 20), int(t - 20))
        larger_box_bottom_right = (int(r + 10), int(b + 10))

        cv2.rectangle(image_with_boxes, larger_box_top_left, larger_box_bottom_right, (0, 0, 255), 2)
        # cv2.putText(image_with_boxes, str(component['Order'][count]), (int(l - 20), int(t - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(image_with_boxes, str(order), (int(l - 20), int(t - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        count = count + 1
    os.makedirs('/home2/sreevatsa/ops_dataset', exist_ok=True)
    output_path = '/home2/sreevatsa/ops_dataset/{}_paragraph_order_interm.png'.format(os.path.basename(image_file_name).split('.')[0])
    cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)) ##


def visualise_para_order(component, image, image_file_name):
    component = component.sort_values(by='Order')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    centers=[]
    for idx, row in component.iterrows():
        top_left = (int(row['Left'][0]), int(row['Top'][1]))
        bottom_right = (int(row['Right'][0]), int(row['Bottom'][1]))
        #get center of the paragraph from top left and bottom right
        center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))
        centers.append(center)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(image, str(row['Order']), (int(row['Left'][0]), int(row['Top'][1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for i in range(1, len(centers)):
        cv2.line(image, centers[i - 1], centers[i], (0, 0, 255), 2)
    # os.makedirs('/home2/sreevatsa/ops_dataset_layoutfilter', exist_ok=True)
    output_path = '/home2/sreevatsa/ops_dataset_layoutfilter/{}_paragraph_order.png'.format(os.path.basename(image_file_name).split('.')[0])
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 

#sort the paragraphs based on top-down and left-right order
def get_paragraph_order(component, image_file_name, output_path):
    '''
    Sorts the paragraphs based on the top-down and left-right order
    '''

    tlbr = []
    for idx, row in component.iterrows():
        tlbr.append([row['Left'][0], row['Top'][1], row['Right'][0], row['Bottom'][1]])
    tlbr_sorted_x = sorted(tlbr, key=lambda x: x[0])
    
    # mean_x = sum([box[0] for box in tlbr_sorted_x]) / len(tlbr_sorted_x)
    # median_x = tlbr_sorted_x[len(tlbr_sorted_x) // 2][0]
    
    mean_width = sum([box[2] - box[0] for box in tlbr_sorted_x]) / len(tlbr_sorted_x)
    median_width = tlbr_sorted_x[len(tlbr_sorted_x) // 2][2] - tlbr_sorted_x[len(tlbr_sorted_x) // 2][0]

    print("Mean width:", mean_width, "Median width:", median_width)
    # mean_x = min(mean_width, median_width)
    mean_x = mean_width
    # mean_x = median_width

    current_vert_line = tlbr_sorted_x[0][0]
    vert_lines = []
    temp_line = []

    for box in tlbr_sorted_x:
        if box[0] >= current_vert_line + (mean_x):
            vert_lines.append(temp_line)
            temp_line = [box]
            current_vert_line = box[0]
            continue
        temp_line.append(box)
    vert_lines.append(temp_line)
    print("Vertical lines")
    print(vert_lines)
    print(len(vert_lines))
    for line in vert_lines:
        line.sort(key=lambda x: x[1])
    
    #visualise the paragraph order
    image = cv2.imread(image_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    #set the order of the paragraphs in component and visualise
    order=0
    for line in vert_lines:
        for box in line:
            for idx, row in component.iterrows():
                if math.ceil(row['Left'][0])== math.ceil(box[0]) and math.ceil(row['Top'][1])== math.ceil(box[1]):
                    component.loc[idx, 'Order'] = order
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(image, str(order), (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    order+=1

                        # Crop the image
                    cropped_image = image[box[1]:box[3], box[0]:box[2]]
                    # Save the cropped image
                    # uncomment the below line to save paragraph level images
                    # cv2.imwrite(f"cropped_{order}.jpg", cropped_image)
                

    # os.makedirs('{}'.format(output_path), exist_ok=True)                
    # output_path = '{}/{}_para_order.png'.format(output_path,os.path.basename(image_file_name).split('.')[0])
    # # output_path = '/home2/sreevatsa/paragraph_order.png'.format(os.path.basename(image_file_name).split('.')[0])
    # cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    #TODO: visualise_para_order() also visualises the paragraph order, so we modify and remove the above code as required

    # print("Component after para order")
    # print(component)  
    

    return component



