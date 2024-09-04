import cv2
from .new_read_order import *

def plot_boxes_in_component(component_df,boxes_df, image_file,flag):
    image = cv2.imread(image_file)
    for index,row in component_df.iterrows():
        coordinates = []
        # print(row)
        box_ids = row['Component'][0]
        for box_id in box_ids:
            coordinates.append(get_TLBR_from_CSV(boxes_df.iloc[box_id]))
        
        # sorted_coordinates = sort_boxes_left_right_top_down(coordinates)
        # print("SORTED COORDINATES",sorted_coordinates)

        #plot the boxes as rectangles and print the order number
        for i,box in enumerate(coordinates):
            # print(box)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imwrite('/home2/sreevatsa/boxes_in_comp_{}.png'.format(flag), image)
    
    # return sorted_coordinates  
    # return coordinates      

