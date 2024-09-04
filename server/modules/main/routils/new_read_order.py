import cv2
import os
import time
import itertools

def get_box_id_from_coordinates(boxes_df, box_coordinates):
    """Get the index of the box in boxes_df based on its coordinates."""
    x1,y1,x2,y2 = box_coordinates
    for index, row in boxes_df.iterrows():
        if (int(row['Top'][1]) == int(y1) and
            int(row['Left'][0]) == int(x1) and
            int(row['Bottom'][1]) == int(y2) and
            int(row['Right'][0]) == int(x2)):
            return index
    return None
    # for index, row in boxes_df.iterrows():
    #     if (row['Top'] == box_coordinates[1] and
    #         row['Left'] == box_coordinates[0] and
    #         row['Bottom'] == box_coordinates[3] and
    #         row['Right'] == box_coordinates[2]):
    #         return index
    # return None

def get_TLBR_from_CSV(df):
    top = df['Top']
    left = df['Left']
    bottom = df['Bottom']
    right = df['Right']
   
    top_left = [int(left[0]), int(top[1])]
    bottom_right = [int(right[0]), int(bottom[1])]
   
    return [top_left[0],top_left[1], bottom_right[0], bottom_right[1]]

# def sort_boxes_left_right_top_down(coordinates):
#     sorted_coordinates = sorted(coordinates, key=lambda x: (x[0], x[1]))
#     return sorted_coordinates


# def sort_bounding_boxes(bboxes):
#     # Step 1: Group bounding boxes by lines
#     lines = []
#     temp_line = []

#     # Sort bounding boxes by their y1 coordinate
#     bboxes.sort(key=lambda x: x[1])

#     for box in bboxes:
#         if not temp_line:
#             temp_line.append(box)
#         else:
#             # Check if y1 of current box is within a threshold of y2 of last box in line
#             if box[1] - temp_line[-1][3] < 15:
#                 temp_line.append(box)
#             else:
#                 lines.append(temp_line)
#                 temp_line = [box]

#     if temp_line:
#         lines.append(temp_line)

#     # Step 2: Sort bounding boxes within each line based on x1 coordinate
#     for line in lines:
#         line.sort(key=lambda x: x[0])

#     # Step 3: Sort lines based on y1 coordinate of first box in each line
#     lines.sort(key=lambda x: x[0][1])

#     # Flatten the sorted lines into a single list
#     sorted_bboxes = [box for line in lines for box in line]

#     return sorted_bboxes

def calculate_median(boxes):
    boxes.sort()
    n = len(boxes)
    if n % 2 == 0:
        return (boxes[n//2] + boxes[n//2 - 1])/2
    else:
        return boxes[n//2]
    
def sort_boxesy(box):
    return box[1]

def sort_boxesx(box):
    return box[0]

def condition_2(box_set, boxst):
    y = boxst[1]
    inter_line_dist = []
    for box in box_set:
        inter_line_dist.append(abs(box[1] - y))
    median_inter_line_dist = calculate_median(inter_line_dist)
    mean_inter_line_dist = sum(inter_line_dist) / len(inter_line_dist)

    print("MEDIAN INTER LINE DIST",median_inter_line_dist)
    print("MEAN INTER LINE DIST",mean_inter_line_dist)

    intra_line_dist = [abs(box1[1] - box2[1]) for box1,box2 in itertools.combinations(box_set, 2)]
    median_intra_line_dist = calculate_median(intra_line_dist)
    mean_intra_line_dist = sum(intra_line_dist) / len(intra_line_dist)

    print("MEDIAN INTRA LINE DIST",median_intra_line_dist)
    print("MEAN INTRA LINE DIST",mean_intra_line_dist)

    # if median_intra_line_dist < median_inter_line_dist:
    if mean_intra_line_dist < mean_inter_line_dist:
        return True
    else:
        return False

def test_sort_words(boxes, image): #IDEA: intra-line box, inter-line box distances and their medians and means
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
    median_height = calculate_median([y2 - y1 for _, y1, _, y2 in boxes])

    print("MEAN HEIGHT",mean_height)
    print("MEDIAN HEIGHT",median_height)
    # boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []

    order=0
    # for expt: to see if the sorted_coordinates_y are correctly in a same line
    # for box in boxes:
    #     order+=1
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    # cv2.imwrite('/home2/sreevatsa/ops/boxes_ordered_test_2.png', image)

    box_set=[]
    for box in boxes:
        
        #check if box is the first box in boxes
        if not box_set:
            # print("BOXSET:", box_set)
            tmp_line.append(box)
            print("box added to tmp_line:", box)
            box_set.append(box)
        elif len(box_set)==1:
            # print("BOXSET:", box_set)
            # print("BOX:", box)
            # print(True if (box[1] >= current_line + median_height) and (condition_2(box_set, box[1])) else False)
            # if (box[1] >= current_line + median_height) and (condition_2(box_set, box[1])):
            # if condition_2(box_set, box[1]):
            if box[1] >= current_line + median_height:    
                # if box[1] >= current_line + median_height:
                lines.append(tmp_line)
                tmp_line = [box]
                current_line = box[1]
                box_set.append(box)
                print("tmp_line",tmp_line)
                continue
            tmp_line.append(box)
            print("box added to tmp_line:", box)
            box_set.append(box)
        elif len(box_set)>1:
            # print("BOXSET:", box_set)
            # print("BOX:", box)
            # print(True if (box[1] >= current_line + median_height) and (condition_2(box_set, box[1])) else False)
            # if (box[1] >= current_line + median_height) and (condition_2(box_set, box[1])):
            # if condition_2(box_set, box[1]):
            if box[1] >= current_line + median_height:
                print("inside condition 1")
                print(box[1] >= current_line + median_height)
                # if box[1] >= current_line + median_height:
                lines.append(tmp_line)
                tmp_line = [box]
                current_line = box[1]
                box_set.append(box)
                print("tmp_line",tmp_line)
                continue
            elif condition_2(box_set, box):
                print("inside condition 2")
                # print(condition_2(box_set, box))
                lines.append(tmp_line)
                tmp_line = [box]
                current_line = box[1]
                box_set.append(box)
                print("tmp_line",tmp_line)
                continue
            else:
                tmp_line.append(box)
                print("box added to tmp_line:", box)
                box_set.append(box)
    lines.append(tmp_line)

    for line in lines:
        line.sort(key=lambda box: box[0])

    return lines

def sort_words(boxes, image): #from Krishna Tulsyan's code
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
    median_height = calculate_median([y2 - y1 for _, y1, _, y2 in boxes])

    # print("MEAN HEIGHT",mean_height)
    # print("MEDIAN HEIGHT",median_height)
    # boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []

    order=0
    # for expt: to see if the sorted_coordinates_y are correctly in a same line
    # for box in boxes:
    #     order+=1
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    # cv2.imwrite('/home2/sreevatsa/ops/boxes_ordered_test_2.png', image)


    for box in boxes:
        # if box[1] > current_line + mean_height:
        # if box[1] >= current_line + (median_height/2):
        if box[1] >= current_line + (mean_height*0.75):
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]
            continue
        tmp_line.append(box)
    lines.append(tmp_line)

    for line in lines:
        line.sort(key=lambda box: box[0])

    return lines


# def get_coordinates_from_component(component_df, boxes_df, image_file):
#     # Filter the DataFrame based on the component_name
#     component_df2 = component_df.iloc[1]
#     image = cv2.imread(image_file)
#     order = 0
    
#     coordinates = []
#     box_ids = component_df2['Component'][0]
#     for box_id in box_ids:
#         coordinates.append(get_TLBR_from_CSV(boxes_df.iloc[box_id]))
#     print("Coordinates: ", coordinates)
#     sorted_coos = sort_words(coordinates)
#     print("Sorted Coordinates: ", sorted_coos)
#     for i,box in enumerate(sorted_coos[0]):
#         order+=1
#         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#         cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
#     # for line in sorted_coos:
#     #     for i,box in enumerate(line):
#     #         order += 1
#     #         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#     #         cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    
#     cv2.imwrite('/home2/sreevatsa/boxes_ordered.png', image)


# def get_coordinates_from_component(component_df,boxes_df, image_file):
#     image = cv2.imread(image_file)
#     order = 0
#     c=0
#     for index,row in component_df.iterrows():
#         c+=1
#         coordinates = []
#         # print(row)
#         box_ids = row['Component'][0]
#         for box_id in box_ids:
#             coordinates.append(get_TLBR_from_CSV(boxes_df.iloc[box_id]))

#         sorted_coordinates_y = sorted(coordinates, key=sort_boxesy)
#         # print("Coordinates: ", sorted_coordinates_y)
#         sorted_coos = sort_words(sorted_coordinates_y)

#         #plot the boxes as rectangles and print the order number
#         # for i,box in enumerate(sorted_coos):
#         for line in sorted_coos:
#             for i,box in enumerate(line):
#                 # print(box)
#                 order += 1
#                 cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                 cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
#     cv2.imwrite('/home2/sreevatsa/boxes_ordered.png', image)
    
    # return sorted_coordinates  
    # return coordinates      

def test_get_coordinates_from_component(component_df, boxes_df, image_file):
    image = cv2.imread(image_file)
    order = 0
    c = 0
    centers = []  # List to store the centers of the boxes
    boxes_df.loc[:,'Visited'] = 0
    boxes_df.loc[:,'Order'] = -1
    for index, row in component_df.iterrows():
        c += 1
        # if c==10 : #for debugging; to print boxes in a certain paragraph; uncomment this line and indent the next following lines untill the loop ends 
        coordinates = []
        box_ids = row['Component'][0]
        for box_id in box_ids:
            coordinates.append(get_TLBR_from_CSV(boxes_df.iloc[box_id]))
        sorted_coordinates_y = sorted(coordinates, key=sort_boxesy)
        # sorted_coordinates_x = sorted(sorted_coordinates_y, key=sort_boxesx)
        # sorted_coordinates_11 = sorted(coordinates, key=lambda x: (x[1], x[0]))
        # print(sorted_coordinates_x, file=open("sorted_coordinates_11.txt", "w"))
        # exit()
        # sorted_coos = test_sort_words(sorted_coordinates_y, image)
        sorted_coos = sort_words(sorted_coordinates_y, image)

        # sort_words(sorted_coordinates_11, image) # for expt: to see if the sorted_coordinates_y are correctly in a same line 
        # break # for expt: to see if the sorted_coordinates_y are correctly in a same line
        
        # sorted_coos1 = sorted_coordinates_x
        # box1 = sorted_coos1[0]
        # cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), 2)
        # print(box1)

        # for expt: to see if the sorted_coordinates_y are correctly in a same line
        # for i, box in enumerate(sorted_coos1):
        #     order += 1
        #     center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        #     centers.append(center)
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        #     print()
        # break

        cc=0
        for line in sorted_coos:
            cc+=1
            for i, box in enumerate(line):
                # print(box)
                # if cc==1: #for debugging; to display boxes in the specified line number only; uncomment this line and the below if block with a break after this loop and indent the next following lines untill the loop ends
                order += 1
                center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)  # Calculate the center of the box
                centers.append(center)  # Add the center to the list
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                
                # Update the Order column in boxes_df with the current order value
                # box_id = get_box_id_from_coordinates(boxes_df, box)
                # boxes_df.at[box_id, 'Visited'] = '1'
                # boxes_df.at[box_id, 'Order'] = order
                # break
            # if cc==1:
            #     break
    # Draw a line between each pair of consecutive centers
    for i in range(1, len(centers)):
        cv2.line(image, centers[i - 1], centers[i], (0, 0, 255), thickness=1)
    os.makedirs('/home2/sreevatsa/ops2', exist_ok=True)
    # cv2.imwrite('/home2/sreevatsa/ops_doclay/boxes_ordered_{}.png'.format(os.path.basename(image_file).split('.')[0]), image)  
    cv2.imwrite('/home2/sreevatsa/ops2/boxes_ordered_{}.png'.format(os.path.basename(image_file).split('.')[0]), image)
    # return image          
        

'''
CURRENTLY WORKING FUNCTION:
 this gets the correct reading order but fails in few cases 
 where the bottom box for a given box is at the height less 
 than the mean height of boxes in a paragraph
'''
# def get_coordinates_from_component(component_df, boxes_df, image_file, output_path):
def get_final_word_order(component_df, boxes_df, image_file, output_path, save_csv):
    image_file_name = image_file.split('/')[-1].split('.')[0]
    image = cv2.imread(image_file)
    order = 0
    c = 0
    centers = []  # List to store the centers of the boxes
    boxes_df.loc[:,'Visited'] = 0
    boxes_df.loc[:,'Order'] = -1
    regions = []
    for index, row in component_df.iterrows():
        c += 1
        coordinates = []
        box_ids = row['Component'][0]
        for box_id in box_ids:
            coordinates.append(get_TLBR_from_CSV(boxes_df.iloc[box_id]))

        sorted_coordinates_y = sorted(coordinates, key=sort_boxesy)
        sorted_coos = sort_words(sorted_coordinates_y, image)

        cc=0
        for line in sorted_coos:
            cc+=1
            for i, box in enumerate(line):
                # print(box)
                box_id = get_box_id_from_coordinates(boxes_df, box)

                if boxes_df.at[box_id, 'Visited'] != 1:
                    order += 1
                    center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)  # Calculate the center of the box
                    centers.append(center)  # Add the center to the list
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=1)
                    cv2.putText(image, str(order), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

                    # Update the Order column in boxes_df with the current order value
                    # box_id = get_box_id_from_coordinates(boxes_df, box)
                    boxes_df.at[box_id, 'Order'] = order
                    boxes_df.at[box_id, 'Visited'] = 1

                    boxwithOrder = {}
                    boxwithOrder['bounding_box'] = {}
                    boxwithOrder['bounding_box']['x'] = box[0]
                    boxwithOrder['bounding_box']['y'] = box[1]
                    boxwithOrder['bounding_box']['w'] = box[2]-box[0] 
                    boxwithOrder['bounding_box']['h'] = box[3]-box[1]
                    boxwithOrder['order'] = order
                    boxwithOrder['line_number'] = cc
                    regions.append(boxwithOrder)

    # Draw a line between each pair of consecutive centers
    for i in range(1, len(centers)):
        cv2.line(image, centers[i - 1], centers[i], (0, 0, 255), thickness=2)
    
    # os.makedirs('{}'.format(output_path), exist_ok=True)
    # cv2.imwrite('{}/boxes_ordered_{}.png'.format(output_path,os.path.basename(image_file).split('.')[0]), image)

    if save_csv == True:
        os.makedirs('/home2/sreevatsa/csv_readingOrder', exist_ok=True)
        boxes_df.to_csv("/home2/sreevatsa/csv_readingOrder/euclidean_{}.csv".format(image_file_name))      
    
    # return boxes_df
    return image, regions