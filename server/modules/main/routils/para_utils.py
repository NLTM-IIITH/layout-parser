from .dist_utils import *
import numpy as np
import cv2

def calculate_bottombox_para(new_df,x):
    bottom = []
    for i in range(len(new_df)):
        dist = []
        id = []
        y_d = []
        for j in range(len(new_df)):
            if i!=j:
                distance = euclidean_distance(np.array(new_df['Top'][j]), np.array(new_df['Bottom'][i]))
                y_distance = abs(new_df['Bottom'][i][1] - new_df['Top'][j][1])
                
                if 0 <= distance < x and i != j:
                    dist.append(distance)
                    id.append(j)
                    y_d.append(y_distance)
        if dist:
            t = np.argmin(y_d)
            d = dist[t]
            bottom.append([d, id[t]])
        else:
            bottom.append([-1, 0])

    new_df['Bottom_Box'] = bottom

#updated for without threshs
def find_closest_paragraphs(df):
    vertical = []
    for index, row in df.iterrows():
        # print("index", index)
        current_box = row[['Top', 'Bottom', 'Left', 'Right']].values
        # print("current box")
        # print(current_box)
        distances_vertical = []
        for other_index, other_row in df.iterrows():
            # print("other_index", other_index)
            if index != other_index:
                other_box = other_row[['Top', 'Bottom', 'Left', 'Right']].values
                distance_top_to_bottom = euclidean_distance1(current_box[1], other_box[0])
                # print("distance_top_to_bottom for ", index, " and ", other_index, " is ", distance_top_to_bottom)
                distance_bottom_to_top = euclidean_distance1(current_box[0], other_box[1])
                # print("distance_bottom_to_top for ", index, " and ", other_index, " is ", distance_bottom_to_top)
                distances_vertical.extend([distance_top_to_bottom, distance_bottom_to_top])
               
        distances_vertical.sort()
        # print("top 3 ", distances_vertical[:3])
        v = sum(distances_vertical[:3])
        t = v/3
        vertical.append(t)

    return vertical

def page_size(image_file):
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    return height, width

def ignore_margins(component, width_p, header, footer, image_file):
    height, width = page_size(image_file)
    top_margin = height*(header/100)
    bottom_margin = height*(footer/100)
    # left_margin = width*(left_m/100)
    # right_margin = width*(right_m/100)
    # vertical_margin = height*(height_p/100)
    horizontal_margin = width*(width_p/100)
    # for i in range(len(component)):
    #     if((component['Top'][i][1] > height - vertical_margin) and len(component['Component'][i][0])<7):
    #         component = component.drop(i)
    #     elif((component['Bottom'][i][1] < vertical_margin) and len(component['Component'][i][0])<7):
    #         component = component.drop(i)
    #     elif(component['Right'][i][0] < horizontal_margin):
    #         component = component.drop(i)
    #     elif(component['Left'][i][0] > width - horizontal_margin):
    #         component = component.drop(i)
    #     else:
    #         continue
    for i in range(len(component)):
        if((component['Top'][i][1] < (top_margin)) and len(component['Component'][i][0])<10):
            component = component.drop(i)
        # elif((component['Bottom'][i][1] > (height - bottom_margin)) and len(component['Component'][i][0])<10):
        #     component = component.drop(i)
        elif((component['Top'][i][1] > (height - bottom_margin)) and len(component['Component'][i][0])<10):
            component = component.drop(i)
        elif(component['Right'][i][0] < horizontal_margin):
            component = component.drop(i)
        elif(component['Left'][i][0] > width - horizontal_margin):
            component = component.drop(i)
        else:
            continue
    return component

def get_next(component, i):
  if(int(component['Bottom_Box'][i][0]) == -1):
    return -1
  else:
    return int(component['Bottom_Box'][i][1])