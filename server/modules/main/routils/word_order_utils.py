import cv2
import numpy as np


def get_next_word(euclidean, i):
    if(int(float(euclidean['Right_Box'][i][0])) == -1):
        return -1
    else:
        return int(float(euclidean['Right_Box'][i][1]))

# def get_next_word(euclidean, i):
#   if(int(float(parse_string(euclidean['Right_Box'][i],'[',','))) == -1):
#     return -1
#   else:
#     return int(float(parse_string(euclidean['Right_Box'][i],',',']')))

# commented below for euclidean
# def minimum_distance(component, euclidean, i):
#     min_x_distance = math.inf
#     min_y_distance = math.inf
#     closest_coordinate = -1

#     for j in component['Component'][i][0]:
#         index = euclidean.index[euclidean['Id'] == j]
#         if euclidean['Visited'].loc[index].values[0] != 1:
#             y_distance = abs(0 - float(euclidean['Top'].loc[index].values[0][1]))
#             x_distance = abs(0 - float(euclidean['Left'].loc[index].values[0][0]))
#             # y_distance = abs(0 - float(parse_string(euclidean['Top'].loc[index].values[0], ',', ']')))
#             # x_distance = abs(0 - float(parse_string(euclidean['Left'].loc[index].values[0], '[', ',')))

#             # if (((y_distance < min_y_distance) or (x_distance < min_x_distance)) or (y_distance == min_y_distance and x_distance < min_x_distance)) and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
#             if (y_distance < min_y_distance or (y_distance == min_y_distance and x_distance < min_x_distance)) and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
#             # if (y_distance < min_y_distance or (y_distance == min_y_distance and x_distance < min_x_distance)) and int(float(parse_string(euclidean['Left_Box'].loc[index].values[0],'[',','))) == -1:
#                 min_x_distance = x_distance
#                 min_y_distance = y_distance
#                 closest_coordinate = j

#     return closest_coordinate

import math

def calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_chebyshev_distance(x1, y1, x2, y2):
    return min(abs(x1 - x2), abs(y1 - y2))

def minimum_distance(component, euclidean, i):
    min_euclidean_distance = math.inf
    min_y_distance = math.inf
    closest_coordinate = -1

    for j in component['Component'][i][0]:
        index = euclidean.index[euclidean['Id'] == j]
        if euclidean['Visited'].loc[index].values[0] != 1:
            x_coordinate = float(euclidean['Left'].loc[index].values[0][0])
            y_coordinate = float(euclidean['Top'].loc[index].values[0][1])

            euclidean_distance = calculate_euclidean_distance(0, 0, x_coordinate, y_coordinate)
            # euclidean_distance = calculate_chebyshev_distance(0, 0, x_coordinate, y_coordinate)

            # if euclidean_distance <= min_euclidean_distance and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
            # if (euclidean_distance < min_euclidean_distance) or (euclidean_distance == min_euclidean_distance and y_coordinate < min_y_distance) and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
            # if (euclidean_distance < min_euclidean_distance and y_coordinate < min_y_distance) or (euclidean_distance == min_euclidean_distance and y_coordinate < min_y_distance) and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
            if (euclidean_distance <= min_euclidean_distance and y_coordinate < min_y_distance) and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
                min_y_distance = y_coordinate
                min_euclidean_distance = euclidean_distance
                closest_coordinate = j
            # if y_coordinate < min_y_coordinate:
            #     min_y_coordinate = y_coordinate
            #     min_euclidean_distance = calculate_euclidean_distance(0, 0, x_coordinate, y_coordinate)
            #     closest_coordinate = j
            # elif y_coordinate == min_y_coordinate:
            #     euclidean_distance = calculate_euclidean_distance(0, 0, x_coordinate, y_coordinate)
            #     if euclidean_distance < min_euclidean_distance:
            #         min_euclidean_distance = euclidean_distance
            #         closest_coordinate = j

    return closest_coordinate

# def minimum_distance(component, euclidean, i):
#     min_euclidean_distance = math.inf
#     min_y_coordinate = math.inf
#     min_x_coordinate = math.inf
#     closest_coordinate = -1

#     # Sort lines based on y-coordinate
#     lines = component['Component'][i][0]
#     sorted_lines = sorted(lines, key=lambda j: float(euclidean['Top'].loc[euclidean['Id'] == j].values[0][1]))

#     for j in sorted_lines:
#         index = euclidean.index[euclidean['Id'] == j]
#         if euclidean['Visited'].loc[index].values[0] != 1:
#             x_coordinate = float(euclidean['Left'].loc[index].values[0][0])
#             y_coordinate = float(euclidean['Top'].loc[index].values[0][1])

#             euclidean_distance = calculate_euclidean_distance(0, 0, x_coordinate, y_coordinate)

#             if euclidean_distance <= min_euclidean_distance and int(float(euclidean['Left_Box'].loc[index].values[0][0])) == -1:
#                 min_euclidean_distance = euclidean_distance
#                 min_y_coordinate = y_coordinate
#                 min_x_coordinate = x_coordinate
#                 closest_coordinate = j

#     return closest_coordinate


def calculate_next_right(component, min_idx, euclidean, i):
    min_x_distance = math.inf
    min_y_distance = math.inf
    closest_coordinate = -1
    index_min = euclidean.index[euclidean['Id'] == min_idx].tolist()[0]
    component_list = component['Component'][i][0]
    for j in component_list:
        index = euclidean.index[euclidean['Id'] == j].tolist()[0]
        if euclidean.at[index, 'Visited'] != 1:
            x_distance = abs(float(euclidean.at[index_min, 'Right'][0] - euclidean.at[index, 'Left'][0]))
            y_distance = abs(float(euclidean.at[index_min, 'Right'][1] - euclidean.at[index, 'Left'][1]))
            # x_distance = abs(float(parse_string(euclidean.at[index_min, 'Right'],'[',',')) - float(parse_string(euclidean.at[index, 'Left'],'[',',')))
            # y_distance = abs(float(parse_string(euclidean.at[index_min, 'Right'],',',']')) - float(parse_string(euclidean.at[index, 'Left'],',',']')))
            if x_distance < min_x_distance and y_distance <=15:
              min_x_distance = x_distance
              min_y_distance = y_distance
              closest_coordinate = j
    return closest_coordinate

def word_order(component, euclidean):
    euclidean.loc[:, 'Order'] = -1
    euclidean.loc[:, 'Visited'] = 0
    euclidean['LineNumber'] = -1
    order = 0
    line_number = 0
    for i in range(len(component)):
        min_idx = minimum_distance(component, euclidean, i)
        visited_list = [euclidean['Visited'][idx] == 0 for idx in component['Component'][i][0]]
        while any(visited_list) != 0 and min_idx != -1:
            if euclidean.at[min_idx, 'Visited'] != 1:
                euclidean.at[min_idx, 'Visited'] = 1
                euclidean.at[min_idx, 'Order'] = order
                euclidean.at[min_idx,'LineNumber'] = line_number
                order += 1
            next_idx = get_next_word(euclidean, min_idx)
            if next_idx != -1:
                min_idx = next_idx
            elif calculate_next_right(component, min_idx, euclidean, i) != -1:
                min_idx = calculate_next_right(component, min_idx, euclidean, i)
            else:
                min_idx = minimum_distance(component, euclidean, i)
                line_number+=1
    return euclidean

def reading_order(image,euclidean, image_file, header_p, footer_p):

    # reading_order_json = {}
    # reading_order_json['image_name'] = image_file
    # reading_order_json['regions'] = []
    regions=[]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_boxes = image_rgb.copy()
    for index, row in euclidean.iterrows():
        left = int(row['Left'][0])
        right = int(row['Right'][0])
        top = int(row['Top'][1])
        bottom = int(row['Bottom'][1])
        # left = int(parse_string(row['Left'],'[',','))
        # right = int(parse_string(row['Right'],'[',','))
        # top = int(parse_string(row['Top'],',',']'))
        # bottom = int(parse_string(row['Bottom'],',',']'))
        Order = int(row['Order'])
        line_number = int(row['LineNumber'])

        width = right - left
        height = bottom - top

        top_left = (left, top)
        bottom_right = (right, bottom)
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (255, 0, 0), 2)

        label_position = (left, top - 10)
        label_position_2 = (left + 40, top - 10)
        cv2.putText(image_with_boxes, str(Order), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.putText(image_with_boxes, str(line_number), label_position_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        image_height, image_width, _ = image_with_boxes.shape
        header_percentage = header_p 
        footer_percentage = footer_p

        header_position = int(image_height * (header_percentage / 100))
        footer_position = image_height - int(image_height * (footer_percentage / 100))

        cv2.line(image_with_boxes, (0, header_position), (image_width, header_position), (255,255, 0), 2)  # yellow line
        cv2.line(image_with_boxes, (0, footer_position), (image_width, footer_position), (255, 255, 0), 2)  
        
        boxwithOrder = {}
        boxwithOrder['bounding_box'] = {}
        boxwithOrder['bounding_box']['x'] = left
        boxwithOrder['bounding_box']['y'] = top
        boxwithOrder['bounding_box']['w'] = width
        boxwithOrder['bounding_box']['h'] = height
        boxwithOrder['order'] = Order
        boxwithOrder['line'] = line_number
        # reading_order_json['regions'].append(boxwithOrder)
        regions.append(boxwithOrder)

    # return image_with_boxes, reading_order_json
    return image_with_boxes, regions


def reading_order_with_line(image, euclidean, image_file, header_p, footer_p):
    regions = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_boxes = image_rgb.copy()

    # Store centers of bounding boxes
    centers = []

    for index, row in euclidean.iterrows():
        left = int(row['Left'][0])
        right = int(row['Right'][0])
        top = int(row['Top'][1])
        bottom = int(row['Bottom'][1])

        Order = int(row['Order'])
        line_number = int(row['LineNumber'])

        width = right - left
        height = bottom - top

        top_left = (left, top)
        bottom_right = (right, bottom)
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (255, 0, 0), 2)

        # Calculate the center of the bounding box
        center_x = left + width // 2
        center_y = top + height // 2
        centers.append((center_x, center_y))

        label_position = (left, top - 10)
        label_position_2 = (left + 40, top - 10)
        cv2.putText(image_with_boxes, str(Order), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        box_with_order = {}
        box_with_order['bounding_box'] = {}
        box_with_order['bounding_box']['x'] = left
        box_with_order['bounding_box']['y'] = top
        box_with_order['bounding_box']['w'] = width
        box_with_order['bounding_box']['h'] = height
        box_with_order['order'] = Order
        box_with_order['line'] = line_number
        regions.append(box_with_order)

    # Draw thick lines connecting the centers of bounding boxes
    thickness = 5
    color = (0, 255, 0)  # Green color
    for i in range(len(centers) - 1):
        if regions[i]['order'] != -1 and regions[i + 1]['order'] != -1:
            cv2.line(image_with_boxes, centers[i], centers[i + 1], color, thickness)

    # for i in range(len(centers) - 1):
    #     cv2.line(image_with_boxes, centers[i], centers[i + 1], color, thickness)

    # Draw header and footer lines
    image_height, image_width, _ = image_with_boxes.shape
    header_percentage = header_p
    footer_percentage = footer_p
    header_position = int(image_height * (header_percentage / 100))
    footer_position = image_height - int(image_height * (footer_percentage / 100))
    cv2.line(image_with_boxes, (0, header_position), (image_width, header_position), (255, 255, 0), 2)
    cv2.line(image_with_boxes, (0, footer_position), (image_width, footer_position), (255, 255, 0), 2)

    return image_with_boxes, regions
