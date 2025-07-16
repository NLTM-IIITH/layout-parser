from .para_utils import *
from .dist_utils import *
import cv2
import numpy as np
import pandas as pd

#ckp3
def column_order(component):
    order = 0
    min_idx = minimum_euclidean(component)

    while any(component['Visited'] == 0) and min_idx != -1:
        if component['Visited'][min_idx] != 1:
            # component['Visited'][min_idx] = 1
            # component['Order'][min_idx] = order
            component.loc[min_idx, 'Visited'] = 1
            component.loc[min_idx, 'Order'] = order
            # order += 1

        next_idx = get_next(component, min_idx)
        if next_idx != -1:
            min_idx = next_idx
        else:
            min_idx = minimum_euclidean(component)
            order+=1
    return component

def merge_columns(image,component):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_boxes = image_rgb.copy()
    df = component
    grouped = df.groupby("Order")
    # Sort each group by "Id" for clarity
    sorted_groups = [group.sort_values(by="Id") for _, group in grouped]
    # Concatenate the sorted groups back into a single DataFrame
    result_df = pd.concat(sorted_groups, ignore_index=True)
    
    # aggregated_df = result_df.groupby("Order")["Component"].apply(lambda x: [item for sublist in x for item in sublist])

    # merged_df = aggregated_df.reset_index()
    # print(result_df,file=open("nmmaraesult.txt","a"))
    print(result_df)
    # for i in range(len(set(np.array(result_df['Order'])))):
    #     same_col=[]
    #     t=[]
    #     l=[]
    #     b=[]
    #     r=[]
    #     for idx, rows in result_df.iterrows():    
    #         if rows['Order'] == i:
    #             same_col.append(rows['Component'][0])
    #             if(int(round(float(rows['Left'][0])))!=-1):
    #                 l.append(int(round(float(rows['Left'][0]))))
    #             if(int(round(float(rows['Right'][0])))!=-1):
    #                 r.append(int(round(float(rows['Right'][0]))))
    #             if(int(round(float(rows['Top'][1])))!=-1):
    #                 t.append(int(round(float(rows['Top'][1]))))
    #             if(int(round(float(rows['Bottom'][1])))!=-1):
    #                 b.append(int(round(float(rows['Bottom'][1]))))
    #     tt = min(t)
    #     bb = max(b)
    #     ll = min(l)
    #     rr = max(r)
    #     center_top = [int(ll+rr)/2, int(tt)]
    #     center_bottom = [int(ll+rr)/2, int(bb)]
    #     center_right = [int(rr), int(tt+bb)/2]
    #     center_left = [int(ll), int(tt+bb)/2]
    #     larger_box_top_left = (int(ll - 20), int(tt - 20))
    #     larger_box_bottom_right = (int(rr + 10), int(bb + 10))
    #     cv2.rectangle(image_with_boxes, larger_box_top_left, larger_box_bottom_right, (0, 0, 255), 2)    
    #     cv2.putText(image_with_boxes, str(i), (int(ll - 20), int(tt - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # # return image_with_boxes
    # output_path = 'column_order.png'
    # cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

# def get_column_order(image,component):
#     comp = component
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_with_boxes = image_rgb.copy()
#     column_order=0
#     for i in range(len(comp)):
#         for j in range(len(comp)):
#             if i!=j:
#                 # print(comp['Bottom'][i][1],comp['Top'][j][1])
#                 dist = np.abs(comp['Bottom'][i][1] - comp['Top'][j][1])
#                 print(dist)
#                 if dist < 250:
#                     #make both the order IDs same
#                     # comp['Order'][i] = 0
#                     # comp['Order'][i]=comp['Order'][j]
#                     # comp['Order'][i] = comp['Order'][j]
#                     # comp.replace(comp['Order'][i],comp['Order'][j],inplace=True)
#                     comp['Order'][i] = column_order
#                 else:
#                     column_order+=1
#                     comp['Order'][i] = column_order    
#     result_df = comp  
#     # print(result_df,file=open("m121mmaraesult.txt","a"))
#     print(result_df)   

#     for i in range(len(set(np.array(result_df['Order'])))):
#         same_col=[]
#         t=[]
#         l=[]
#         b=[]
#         r=[]
#         for idx, rows in result_df.iterrows():    
#             if rows['Order'] == i:
#                 same_col.append(rows['Component'][0])
#                 if(int(round(float(rows['Left'][0])))!=-1):
#                     l.append(int(round(float(rows['Left'][0]))))
#                 if(int(round(float(rows['Right'][0])))!=-1):
#                     r.append(int(round(float(rows['Right'][0]))))
#                 if(int(round(float(rows['Top'][1])))!=-1):
#                     t.append(int(round(float(rows['Top'][1]))))
#                 if(int(round(float(rows['Bottom'][1])))!=-1):
#                     b.append(int(round(float(rows['Bottom'][1]))))
#         tt = min(t)
#         bb = max(b)
#         ll = min(l)
#         rr = max(r)
#         center_top = [int(ll+rr)/2, int(tt)]
#         center_bottom = [int(ll+rr)/2, int(bb)]
#         center_right = [int(rr), int(tt+bb)/2]
#         center_left = [int(ll), int(tt+bb)/2]
#         larger_box_top_left = (int(ll - 20), int(tt - 20))
#         larger_box_bottom_right = (int(rr + 10), int(bb + 10))
#         cv2.rectangle(image_with_boxes, larger_box_top_left, larger_box_bottom_right, (0, 0, 255), 2)    
#         cv2.putText(image_with_boxes, str(i), (int(ll - 20), int(tt - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     # return image_with_boxes
#     output_path = 'column_order.png'
#     cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))                


def get_col(image,comp):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_boxes = image_rgb.copy()

    df = comp
    grouped = df.groupby("Order")
    # Sort each group by "Id" for clarity
    sorted_groups = [group.sort_values(by="Id") for _, group in grouped]
    # Concatenate the sorted groups back into a single DataFrame
    result_df = pd.concat(sorted_groups, ignore_index=True)

    component = result_df
    col_n=0
    component['Col'] = 0
    
    for i in range(len(component)-1):
        print(component['Bottom'][i][1],component['Top'][i+1][1])
        dist = np.abs(component['Bottom'][i][1] - component['Top'][i+1][1])
        print(dist)
        # if dist < 500:
        if component['Bottom'][i][1] < component['Top'][i+1][1] and(dist < 200):
            component['Col'][i] = col_n
            component['Col'][i+1] = col_n
        else:
            col_n+=1
            component['Col'][i+1] = col_n    
    print(component)

    result_df = component  
    for i in range(len(set(np.array(result_df['Col'])))):
        same_col=[]
        t=[]
        l=[]
        b=[]
        r=[]
        for idx, rows in result_df.iterrows():    
            if rows['Col'] == i:
                same_col.append(rows['Component'][0])
                if(int(round(float(rows['Left'][0])))!=-1):
                    l.append(int(round(float(rows['Left'][0]))))
                if(int(round(float(rows['Right'][0])))!=-1):
                    r.append(int(round(float(rows['Right'][0]))))
                if(int(round(float(rows['Top'][1])))!=-1):
                    t.append(int(round(float(rows['Top'][1]))))
                if(int(round(float(rows['Bottom'][1])))!=-1):
                    b.append(int(round(float(rows['Bottom'][1]))))
        tt = min(t)
        bb = max(b)
        ll = min(l)
        rr = max(r)
        center_top = [int(ll+rr)/2, int(tt)]
        center_bottom = [int(ll+rr)/2, int(bb)]
        center_right = [int(rr), int(tt+bb)/2]
        center_left = [int(ll), int(tt+bb)/2]
        larger_box_top_left = (int(ll - 20), int(tt - 20))
        larger_box_bottom_right = (int(rr + 10), int(bb + 10))
        cv2.rectangle(image_with_boxes, larger_box_top_left, larger_box_bottom_right, (0, 0, 255), 2)    
        cv2.putText(image_with_boxes, str(i), (int(ll - 20), int(tt - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # return image_with_boxes
    output_path = 'column_order.png'
    # cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))  ##

#ckp4 -> ckp3 to ck4 for column order; TODO: fix column order working, currently in a very rough state, does not work well for all images!
