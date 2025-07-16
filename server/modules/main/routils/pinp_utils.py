from .para_utils import *
from .kde_utils import *
from .dist_utils import *
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# VISULISATION
def draw_rectangles_paras_cv2(df, image_filename, id):
    # Create a blank white image
    img = cv2.imread(image_filename)

    for index, row in df.iterrows():
        top = row['Top']
        bottom = row['Bottom']
        left = row['Left']
        right = row['Right']

        top_left = (left[0], top[1])
        bottom_right = (right[0], bottom[1])
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        # cv2.putText(img, str(row['Id']), (left[0], top[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str(index), (left[0], top[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite('/home2/sreevatsa/rectangles_{}.png'.format(id), img)

def calculate_overlap_percentage(large_box, small_box):

    large_x1, large_y1, large_x2, large_y2 = large_box
    small_x1, small_y1, small_x2, small_y2 = small_box

    overlap_x1 = max(large_x1, small_x1)
    overlap_y1 = max(large_y1, small_y1)
    overlap_x2 = min(large_x2, small_x2)
    overlap_y2 = min(large_y2, small_y2)

    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    large_area = (large_x2 - large_x1) * (large_y2 - large_y1)
    small_area = (small_x2 - small_x1) * (small_y2 - small_y1)

    # overlap_percentage = (overlap_area / large_area) * 100
    overlap_percentage = (overlap_area/small_area)*100
    return overlap_percentage

def is_box_inside(large_box, small_box, threshold_percentage):

    large_x1, large_y1, large_x2, large_y2 = large_box
    small_x1, small_y1, small_x2, small_y2 = small_box

    overlap_percentage = calculate_overlap_percentage(large_box, small_box)
    # print("Overlap:",overlap_percentage)
    if (large_x1 < small_x1 and small_x2 < large_x2 and large_y1 < small_y1 and small_y2 < large_y2):
        return True
    elif (overlap_percentage >= threshold_percentage):
        return True
    else:
      return False


def pinp(component, width_p, header_p, footer_p, image_filename, theta):
    df = pd.DataFrame(component)
    df['Merged'] = False
    filtered_df_1 = df[df['Component'].apply(lambda x: len(x[0]) >theta)]
    filtered_df_2 = df[df['Component'].apply(lambda x: len(x[0]) <=theta)]
    # filtered_df_1 = df.copy()
    # filtered_df_2 = df.copy()

    # print("filtered_df_2", filtered_df_2)
    # print("filtered_df_1", filtered_df_1)

    # @title Default title text
    df2 = df.copy()

    if not filtered_df_2.empty:  # Check if filtered_df_2 is not empty
        for i1, r1 in filtered_df_1.iterrows():
            # print("i1", i1)
            temp_new_list=[]
            for i2, r2 in filtered_df_2.iterrows():
                if r1.equals(r2) == False and r2['Merged'] == False:
                    # print("i2", i2)

                    #getting top left and bottom right coordinates of the boxes for filtered_df_1
                    tl1 = [r1['Left'][0], r1['Top'][1]]
                    br1 = [r1['Right'][0], r1['Bottom'][1]]
                    tlbr1 = [tl1[0], tl1[1], br1[0], br1[1]]

                    #getting top left and bottom right coordinates of the boxes for filtered_df_2
                    tl2 = [r2['Left'][0], r2['Top'][1]]
                    br2 = [r2['Right'][0], r2['Bottom'][1]]
                    tlbr2 = [tl2[0], tl2[1], br2[0], br2[1]]
                    
                    # print("tlbr1", tlbr1, "tlbr2", tlbr2)

                    if is_box_inside(tlbr1, tlbr2, threshold_percentage=85) == False:
                        new_list = [r1['Component'][0]]               
                    # elif is_box_inside(tlbr1, tlbr2, threshold_percentage=85) == True:
                    else:    
                        temp_new_list.append(r2['Component'][0])
                        filtered_df_2.at[i2, 'Merged'] = True
                        df2.at[i2, 'Merged'] = True
            temp_temp = [item for sublist in temp_new_list for item in sublist]
            new_list = [r1['Component'][0] + temp_temp]
            df2.at[i1, 'Component'] = new_list
            # print("NEWLIST",new_list)
        # print('df2')
        # print(df2)      

        filtered_dfffg = df2[df2['Merged'] == False].reset_index(drop=True)    
        # filtered_dfffg = df2[df2['Component'].apply(lambda x: len(x[0]) >= theta)].reset_index(drop=True)
    
        filtered_dfffg['Id'] = range(0, len(filtered_dfffg))
        filtered_dfffg['Bottom_Box'] = filtered_dfffg['Bottom_Box'].apply(lambda x: [-1, 0])
        filtered_dfffg = ignore_margins(filtered_dfffg, width_p, header_p, footer_p, image_filename)
        filtered_dfffg = filtered_dfffg.reset_index(drop=True)

        
        '''
        considering and adding back the paragraphs that are not inside bigger paras but are in between the bigger paras 
        while satisfying the number of words in a para condition. 
        This is done to make sure that the paragraphs that that are in between the bigger paras are not missed out.
        The previous implementation was missing out the paragraphs that are in between the bigger paras. So added the below for loop.
        '''
        # for i2, r2 in filtered_df_2.iterrows():
        #     if not any((filtered_dfffg['Component'].apply(lambda x: r2['Component'][0] in x))):
        #         filtered_dfffg = filtered_dfffg.append(r2, ignore_index=True)


        # print("filtered_dfffg after pinp but not ordered before find_closest_paragraphs and kde_estimate and calculate_bottombox_para") 
        # print(filtered_dfffg)

        # if len(filtered_dfffg) >= 3:
        #     vertical = find_closest_paragraphs(filtered_dfffg)
        #     print("VERTICAL", vertical)
        #     x_para_1 = kde_estimate(vertical)
        #     x_para_2 = np.mean(vertical)
        #     print("x_para_1", x_para_1, "x_para_2", x_para_2)
        #     x_para = min(x_para_1, x_para_2)
        #     calculate_bottombox_para(filtered_dfffg, x_para)

        #     return filtered_dfffg
        # else:
        #     return filtered_dfffg
        
        '''
        all the paragraphs irrespective of the number of paras after the pinp fix, 
        should have to go through the finding_closest_paragraphs, kde_estimate and 
        calculate_bottombox_para
        '''
        vertical = find_closest_paragraphs(filtered_dfffg)
        # print("VERTICAL", vertical)
        x_para_1 = kde_estimate(vertical)
        x_para_2 = np.mean(vertical)
        # print("x_para_1", x_para_1, "x_para_2", x_para_2)
        x_para = min(x_para_1, x_para_2)
        calculate_bottombox_para(filtered_dfffg, x_para)

        component_after_pinp_not_ordered = filtered_dfffg

        # #adding back the paragraphs that are not inside bigger paras but are in between the bigger paras
        # # Create a new DataFrame with paragraphs from filtered_df_2 that do not satisfy the is_box_inside condition
        # filtered_df2_remaining = filtered_df_2[~filtered_df_2.index.isin(filtered_dfffg.index)]

        # # Combine filtered_dfffg with the remaining paragraphs from filtered_df_2
        # combined_df = pd.concat([filtered_dfffg, filtered_df2_remaining]).reset_index(drop=True)

        
        # component_after_pinp_not_ordered = combined_df 
    
        return component_after_pinp_not_ordered
    else:
        # Handle the case when filtered_df_2 is empty
        print("filtered_df_2 is empty")
        return df



# def pinp(component,width_p, header_p, footer_p, image_filename):
#     df = pd.DataFrame(component)
#     filtered_df_2 = df[df['Component'].apply(lambda x: len(x[0]) < 6)]
#     filtered_df_1 = df[df['Component'].apply(lambda x: len(x[0]) >= 6)]

#     print("filtered_df_2")
#     print(filtered_df_2)
#     print("filtered_df_1")
#     print(filtered_df_1)

#     # @title Default title text
#     df2 = df.copy()
#     for i1, r1 in filtered_df_1.iterrows():
#         new_list=[]
#         for i2,r2 in filtered_df_2.iterrows():
#             if r1.equals(r2) == False:
#                 tl1 = [r1['Left'][0], r1['Top'][1]]
#                 br1 = [r1['Right'][0], r1['Bottom'][1]]
#                 tlbr1 = [tl1[0], tl1[1], br1[0], br1[1]]

#                 tl2 = [r2['Left'][0], r2['Top'][1]]
#                 br2 = [r2['Right'][0], r2['Bottom'][1]]
#                 tlbr2 = [tl2[0], tl2[1], br2[0], br1[1]]

#                 # print("before condition")
#                 # print(r1['Component'])
#                 # print(r2['Component'])

#                 if is_box_inside(tlbr1, tlbr2, threshold_percentage=85) == False:
#                     new_list = [r1['Component'][0]]
#                 elif is_box_inside(tlbr1, tlbr2, threshold_percentage=85) == True:
#                     # print("inside condition")
#                     # print(r1['Component'])
#                     # print(r2['Component'])

#                     new_list = [r1['Component'][0] + r2['Component'][0]]
#                     # print((new_list))
#                     # print(i1)
#                     r1['Component'] = new_list
                
#         df2.at[i1,'Component'] = new_list
#         # print("round of smalls done")

#         # df2.drop(i2,inplace=False)

#     print("GGGG")
#     print(df2)
#     # filtered_dfffg = df2[df2['Component'].apply(lambda x: len(x[0]) >= 6)].reset_index(drop=True)
#     filtered_dfffg = df2[df2['Component'].apply(lambda x: len(x[0]) >= 6)].reset_index(drop=True)
#     # filtered_dfffg = filtered_dfffg.reset_index().rename(columns={'index': 'Id'})
#     filtered_dfffg['Id'] = range(0, len(filtered_dfffg))

#     #update all rows of filtered_dfffg['Bottom Box'] to [-1,0]
#     filtered_dfffg['Bottom_Box'] = filtered_dfffg['Bottom_Box'].apply(lambda x: [-1,0])

#     filtered_dfffg = ignore_margins(filtered_dfffg,width_p,header_p,footer_p,image_filename)
#     filtered_dfffg = filtered_dfffg.reset_index(drop=True)

#     if len(filtered_dfffg) >=3:

#         vertical = find_closest_paragraphs(filtered_dfffg) #added
#         print("VETICAL",vertical)
#         x_para = kde_estimate(vertical) #added
#         calculate_bottombox_para(filtered_dfffg,x_para) #added

#         return filtered_dfffg
#     else:
#         return filtered_dfffg


def pinp2(component, width_p, header_p, footer_p, image_filename, theta):
    df = pd.DataFrame(component)
    df['Merged'] = False
    filtered_df_1 = df.copy()

    # print("filtered_df_1", filtered_df_1)

    # @title Default title text
    df2 = df.copy()

    for i1, r1 in filtered_df_1.iterrows():
        
        # print('i1', i1, 'Merged', filtered_df_1.iloc[i1]['Merged'])
        temp_new_list=[]
        if filtered_df_1.iloc[i1]['Merged'] == False:
            for i2, r2 in filtered_df_1.iterrows():
                if r1.equals(r2) == False and r2['Merged'] == False:
                    print("i2", i2)

                    #getting top left and bottom right coordinates of the boxes for filtered_df_1
                    tl1 = [r1['Left'][0], r1['Top'][1]]
                    br1 = [r1['Right'][0], r1['Bottom'][1]]
                    tlbr1 = [tl1[0], tl1[1], br1[0], br1[1]]

                    #getting top left and bottom right coordinates of the boxes for filtered_df_2
                    tl2 = [r2['Left'][0], r2['Top'][1]]
                    br2 = [r2['Right'][0], r2['Bottom'][1]]
                    tlbr2 = [tl2[0], tl2[1], br2[0], br2[1]]

                    if is_box_inside(tlbr1, tlbr2, threshold_percentage=10) == False:
                        new_list = [r1['Component'][0]] 
                    else:    
                        temp_new_list.append(r2['Component'][0])
                        # print("temp_new_list", temp_new_list)
                        filtered_df_1.at[i2, 'Merged'] = True
                        df2.at[i2, 'Merged'] = True

                        #merging and extending overlapping paragraphs
                        merged_tlbr = [min(tlbr1[0], tlbr2[0]), min(tlbr1[1], tlbr2[1]), max(tlbr1[2], tlbr2[2]), max(tlbr1[3], tlbr2[3])]
                        updated_center_top = [merged_tlbr[0] + (merged_tlbr[2] - merged_tlbr[0]) / 2, merged_tlbr[1]]
                        updated_center_bottom = [merged_tlbr[0] + (merged_tlbr[2] - merged_tlbr[0]) / 2, merged_tlbr[3]]
                        updated_center_left = [merged_tlbr[0], merged_tlbr[1] + (merged_tlbr[3] - merged_tlbr[1]) / 2]
                        updated_center_right = [merged_tlbr[2], merged_tlbr[1] + (merged_tlbr[3] - merged_tlbr[1]) / 2]

                        df2.at[i1, 'Top'] = updated_center_top
                        df2.at[i1, 'Bottom'] = updated_center_bottom
                        df2.at[i1, 'Left'] = updated_center_left
                        df2.at[i1, 'Right'] = updated_center_right
        temp_temp = [item for sublist in temp_new_list for item in sublist]
        # print("temp_temp", temp_temp)
        new_list = [r1['Component'][0] + temp_temp]
        df2.at[i1, 'Component'] = new_list
        # print("NEWLIST",new_list)

        
        
        
        # print('df2')
        # print(df2)
    
    filtered_dfffg = df2[df2['Merged'] == False].reset_index(drop=True)    
    # filtered_dfffg = df2[df2['Component'].apply(lambda x: len(x[0]) >= theta)].reset_index(drop=True)

    filtered_dfffg['Id'] = range(0, len(filtered_dfffg))
    filtered_dfffg['Bottom_Box'] = filtered_dfffg['Bottom_Box'].apply(lambda x: [-1, 0])
    filtered_dfffg = ignore_margins(filtered_dfffg, width_p, header_p, footer_p, image_filename)
    filtered_dfffg = filtered_dfffg.reset_index(drop=True)

            
    '''
    all the paragraphs irrespective of the number of paras after the pinp fix, 
    should have to go through the finding_closest_paragraphs, kde_estimate and 
    calculate_bottombox_para
    '''
    vertical = find_closest_paragraphs(filtered_dfffg)
    # print("VERTICAL", vertical)
    x_para_1 = kde_estimate(vertical)
    x_para_2 = np.mean(vertical)
    # print("x_para_1", x_para_1, "x_para_2", x_para_2)
    x_para = min(x_para_1, x_para_2)
    calculate_bottombox_para(filtered_dfffg, x_para)

    component_after_pinp_not_ordered = filtered_dfffg

    return component_after_pinp_not_ordered

def merge_dangling(component, width_p, header_p, footer_p, image_filename, theta):
    df = pd.DataFrame(component)
    filtered_df_1 = df[df['Component'].apply(lambda x: len(x[0]) >theta)]
    filtered_df_2 = df[df['Component'].apply(lambda x: len(x[0]) <=theta)]

    #find nearest paragraph for each of filtered_df_2 and merge them
    for i1, r1 in filtered_df_2.iterrows():
        for i2, r2 in filtered_df_1.iterrows():
            if r1.equals(r2) == False:
                top_r1 = r1['Top']
                bottom_r1 = r1['Bottom']
                
                top_r2 = r2['Top']
                bottom_r2 = r2['Bottom']

                #find closest r1 to r2
                distance_top_to_bottom = euclidean_distance1(bottom_r1, top_r2)
                distance_bottom_to_top = euclidean_distance1(top_r1, bottom_r2)

                #get nearest box index
                


    filtered_dfffg = df[df['Merged'] == False].reset_index(drop=True)
    filtered_dfffg['Id'] = range(0, len(filtered_dfffg))
    filtered_dfffg['Bottom_Box'] = filtered_dfffg['Bottom_Box'].apply(lambda x: [-1, 0])
    filtered_dfffg = ignore_margins(filtered_dfffg, width_p, header_p, footer_p, image_filename)
    filtered_dfffg = filtered_dfffg.reset_index(drop=True)

    return filtered_dfffg


