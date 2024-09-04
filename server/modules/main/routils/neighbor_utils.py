from .kde_utils import *
import cv2
from .global_utils import args
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cdist
from .dist_utils import *


def calculate_center_points(df, new_df):
    top = []
    bottom = []
    left = []
    right = []
    id = []

    for i in range(0,len(df)):
        x1,y1,x2,y2 = df[0][i][0], df[0][i][1], df[0][i][2], df[0][i][3]
        top_center = [(x1 + x2) / 2, y1]
        bottom_center = [(x1 + x2) / 2, y2]
        left_center = [x1, (y1 + y2) / 2]
        right_center = [x2, (y1 + y2) / 2]
        top.append(top_center)
        bottom.append(bottom_center)
        left.append(left_center)
        right.append(right_center)
        id.append(i)
    new_df['Top'] = top
    new_df['Bottom'] = bottom
    new_df['Left'] = left
    new_df['Right'] = right
    new_df['Id'] = id



# def find_closest_neighbors(df):
#     horizontal = []
#     vertical = []
#     for index, row in df.iterrows():
#         current_box = row[['Top', 'Bottom', 'Left', 'Right']].values
#         distances_horizontal = []
#         distances_vertical = []
#         for other_index, other_row in df.iterrows():
#             if index != other_index:
#                 other_box = other_row[['Top', 'Bottom', 'Left', 'Right']].values
#                 distance_left_to_right = euclidean_distance1(current_box[2], other_box[3])
#                 distance_right_to_left = euclidean_distance1(current_box[3], other_box[2])
#                 distances_horizontal.extend([distance_left_to_right, distance_right_to_left])
#         distances_horizontal.sort()
#         s = sum(distances_horizontal[:6])
#         a = s/6
#         horizontal.append(a)

#         for other_index, other_row in df.iterrows():
#             if index != other_index:
#                 other_box = other_row[['Top', 'Bottom', 'Left', 'Right']].values
#                 distance_top_to_bottom = euclidean_distance1(current_box[1], other_box[0])
#                 distance_bottom_to_top = euclidean_distance1(current_box[0], other_box[1])
#                 distances_vertical.extend([distance_top_to_bottom, distance_bottom_to_top])
#         distances_vertical.sort()
#         v = sum(distances_vertical[:2])
#         t = v/2
#         vertical.append(t)

#     return horizontal, vertical




# def find_closest_neighbors(df):
#     horizontal = []
#     vertical = []

#     box_data = df[['Top', 'Bottom', 'Left', 'Right']].values

#     for index, current_box in enumerate(box_data):
#         distances_horizontal = []
#         distances_vertical = []

#         for other_index, other_box in enumerate(box_data):
#             if index != other_index:
#                 distance_left_to_right = euclidean(current_box[2], other_box[3])
#                 distance_right_to_left = euclidean(current_box[3], other_box[2])
#                 distances_horizontal.extend([distance_left_to_right, distance_right_to_left])

#                 distance_top_to_bottom = euclidean(current_box[1], other_box[0])
#                 distance_bottom_to_top = euclidean(current_box[0], other_box[1])
#                 distances_vertical.extend([distance_top_to_bottom, distance_bottom_to_top])

#         distances_horizontal.sort()
#         s = sum(distances_horizontal[:6])
#         horizontal.append(s / 6)

#         distances_vertical.sort()
#         v = sum(distances_vertical[:2])
#         vertical.append(v / 2)

#     return horizontal, vertical
    
def find_closest_neighbors(df):
    horizontal = []
    vertical = []

    box_data = df[['Top', 'Bottom', 'Left', 'Right']].values

    for index, current_box in enumerate(box_data):
        distances_horizontal = []
        distances_vertical = []

        for other_index, other_box in enumerate(box_data):
            if index != other_index:
                if args.metric == 'euclidean':    
                    distance_left_to_right = euclidean(current_box[2], other_box[3])
                    distance_right_to_left = euclidean(current_box[3], other_box[2])
                    distances_horizontal.extend([distance_left_to_right, distance_right_to_left])

                    distance_top_to_bottom = euclidean(current_box[1], other_box[0])
                    distance_bottom_to_top = euclidean(current_box[0], other_box[1])
                    distances_vertical.extend([distance_top_to_bottom, distance_bottom_to_top])
                elif args.metric == 'chebyshev':
                    distance_left_to_right = chebyshev(current_box[2], other_box[3])
                    distance_right_to_left = chebyshev(current_box[3], other_box[2])
                    distances_horizontal.extend([distance_left_to_right, distance_right_to_left])

                    distance_top_to_bottom = chebyshev(current_box[1], other_box[0])
                    distance_bottom_to_top = chebyshev(current_box[0], other_box[1])
                    distances_vertical.extend([distance_top_to_bottom, distance_bottom_to_top])

        distances_horizontal.sort()
        s = sum(distances_horizontal[:3])
        horizontal.append(s / 3)

        distances_vertical.sort()
        v = sum(distances_vertical[:3])
        vertical.append(v / 3)

    return horizontal, vertical


# def calculate_rightbox(new_df,x):
#     right = []
#     for i in range(len(new_df)):
#         dist = []
#         id = []
#         for j in range(len(new_df)):
#             distance = euclidean_distance(np.array(new_df['Left'][j]), np.array(new_df['Right'][i]))
#             y_distance = abs(new_df['Right'][i][1] - new_df['Left'][j][1])
#             if 0 <= distance <= x and i != j and y_distance < 20:
#                 dist.append(distance)
#                 id.append(j)
#         if dist:
#             t = np.argmin(dist)
#             right.append([np.min(dist), id[t]])
#         else:
#             right.append([-1, 0])
#     # print(right)
#     new_df['Right_Box'] = right
    

# def calculate_leftbox(new_df,x):
#     left = []
#     for i in range(len(new_df)):
#         dist = []
#         id = []
#         for j in range(len(new_df)):
#             distance = euclidean_distance(np.array(new_df['Right'][j]), np.array(new_df['Left'][i]))
#             y_distance = abs(new_df['Left'][i][1] - new_df['Right'][j][1])
#             if 0 <= distance < x and i != j and y_distance < 20:
#                 dist.append(distance)
#                 id.append(j)
#         if dist:
#             t = np.argmin(dist)
#             left.append([np.min(dist), id[t]])
#         else:
#             left.append([-1, 0])

#     new_df['Left_Box'] = left

# def calculate_topbox(new_df,y):
#     top = []
#     for i in range(len(new_df)):
#         dist = []
#         id = []
#         for j in range(len(new_df)):
#             distance = euclidean_distance(np.array(new_df['Bottom'][j]), np.array(new_df['Top'][i]))
#             if 0 <= distance < y and i != j:
#             # if 0 <= distance and i != j:
#                 dist.append(distance)
#                 id.append(j)
#         if dist:
#             t = np.argmin(dist)
#             top.append([np.min(dist), id[t]])
#         else:
#             top.append([-1, 0])

#     new_df['Top_Box'] = top

# def calculate_bottombox(new_df,y):
#     bottom = []
#     for i in range(len(new_df)):
#         dist = []
#         id = []
#         for j in range(len(new_df)):
#             distance = euclidean_distance(np.array(new_df['Top'][j]), np.array(new_df['Bottom'][i]))
#             if 0 <= distance < y and i != j:
#             # if 0 <= distance and i != j:
#                 dist.append(distance)
#                 id.append(j)
#         if dist:
#             t = np.argmin(dist)
#             bottom.append([np.min(dist), id[t]])
#         else:
#             bottom.append([-1, 0])

#     new_df['Bottom_Box'] = bottom

#using iterrows()
def calculate_rightbox(new_df,x):
    right = []
    for i, row in new_df.iterrows():
        dist = []
        id = []
        for j, other_row in new_df.iterrows():
            distance = euclidean(row['Left'], other_row['Right'])
            y_distance = abs(row['Right'][1] - other_row['Left'][1])
            if 0 <= distance <= x and i != j and y_distance < 20:
                dist.append(distance)
                id.append(j)
        if dist:
            t = np.argmin(dist)
            right.append([np.min(dist), id[t]])
        else:
            right.append([-1, 0])

    new_df['Right_Box'] = right

def calculate_leftbox(new_df,x):
    left = []
    for i, row in new_df.iterrows():
        dist = []
        id = []
        for j, other_row in new_df.iterrows():
            distance = euclidean(row['Right'], other_row['Left'])
            y_distance = abs(row['Left'][1] - other_row['Right'][1])
            if 0 <= distance < x and i != j and y_distance < 20:
                dist.append(distance)
                id.append(j)
        if dist:
            t = np.argmin(dist)
            left.append([np.min(dist), id[t]])
        else:
            left.append([-1, 0])

    new_df['Left_Box'] = left

def calculate_topbox(new_df,y):
    top = []
    for i, row in new_df.iterrows():
        dist = []
        id = []
        for j, other_row in new_df.iterrows():
            distance = euclidean(row['Bottom'], other_row['Top'])
            if 0 <= distance < y and i != j:
                dist.append(distance)
                id.append(j)
        if dist:
            t = np.argmin(dist)
            top.append([np.min(dist), id[t]])
        else:
            top.append([-1, 0])

    new_df['Top_Box'] = top

def calculate_bottombox(new_df,y):
    bottom = []
    for i, row in new_df.iterrows():
        dist = []
        id = []
        for j, other_row in new_df.iterrows():
            distance = euclidean(row['Top'], other_row['Bottom'])
            if 0 <= distance < y and i != j:
                dist.append(distance)
                id.append(j)
        if dist:
            t = np.argmin(dist)
            bottom.append([np.min(dist), id[t]])
        else:
            bottom.append([-1, 0])

    new_df['Bottom_Box'] = bottom

def make_connections(image, euclidean):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_with_boxes = image_rgb.copy()

    for index, row in euclidean.iterrows():
        left = int(row['Left'][0])
        right = int(row['Right'][0])
        top = int(row['Top'][1])
        bottom = int(row['Bottom'][1])
        box_id = int(row['Id'])

        width = right - left
        height = bottom - top

        top_left = (left, top)
        bottom_right = (right, bottom)

        cv2.rectangle(image_with_boxes, top_left, bottom_right, (255, 0, 0), 2)

        label_position = (left, top - 10)
        cv2.putText(image_with_boxes, str(box_id), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        top_adjacent_id = int(row['Top_Box'][1])
        bottom_adjacent_id = int(row['Bottom_Box'][1])
        left_adjacent_id = int(row['Left_Box'][1])
        right_adjacent_id = int(row['Right_Box'][1])

        if top_adjacent_id != 0:
            top_adjacent_row = euclidean[euclidean['Id'] == top_adjacent_id].iloc[0]
            top_adjacent_center = int(top_adjacent_row['Bottom'][0]) , int(top_adjacent_row['Bottom'][1])
            cv2.line(image_with_boxes, (int(left) + width // 2, int(top)), top_adjacent_center, (0, 255, 0), 2)

        if bottom_adjacent_id != 0:
            bottom_adjacent_row = euclidean[euclidean['Id'] == bottom_adjacent_id].iloc[0]
            bottom_adjacent_center = int(bottom_adjacent_row['Top'][0]) , int(bottom_adjacent_row['Top'][1])
            cv2.line(image_with_boxes, (int(left) + width // 2, int(bottom)), (int(bottom_adjacent_center[0]), int(bottom_adjacent_center[1])), (0, 255, 0), 2)

        if left_adjacent_id != 0:
            left_adjacent_row = euclidean[euclidean['Id'] == left_adjacent_id].iloc[0]
            left_adjacent_center = int(left_adjacent_row['Right'][0]) , int(left_adjacent_row['Right'][1])
            cv2.line(image_with_boxes, (int(left), int(top) + height // 2), (int(left_adjacent_center[0]), int(left_adjacent_center[1])), (0, 255, 0), 2)

        if right_adjacent_id != 0:
            right_adjacent_row = euclidean[euclidean['Id'] == right_adjacent_id].iloc[0]
            right_adjacent_center = int(right_adjacent_row['Left'][0]) , int(right_adjacent_row['Left'][1])
            cv2.line(image_with_boxes, (int(right), int(top) + height // 2), (int(right_adjacent_center[0]), int(right_adjacent_center[1])), (0, 255, 0), 2)

    return image_with_boxes
