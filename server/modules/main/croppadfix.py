import torch
import datetime
from doctr.models import detection
import numpy as np
from PIL import Image
# from torchvision.transforms import Normalize
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
from typing import Any
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from collections import OrderedDict
from doctr.utils.visualization import visualize_page
from datetime import date
import cv2
import os
import json
import re
import shutil
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from .helper import doctr_predictions


today = date.today()
d=today.strftime("%d%m%y")

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%H%M%S")

# # doctr_predictions imported from .helper
# def doctr_predictions_dir(directory): 
#     doc = DocumentFile.from_images(directory)
#     # print('TPYRY')
#     # print(type(doc))
#     # print(np.shape(doc))
#     # print(doc)
#     result = predictor(doc)
#     dic = result.export()
    
#     page_dims = [page['dimensions'] for page in dic['pages']]
#     print(page_dims)
#     regions = []
#     abs_coords = []
    
#     regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
#     abs_coords = [
#     [[int(round(word['geometry'][0][0] * dims[1])), 
#       int(round(word['geometry'][0][1] * dims[0])), 
#       int(round(word['geometry'][1][0] * dims[1])), 
#       int(round(word['geometry'][1][1] * dims[0]))] for word in words]
#     for words, dims in zip(regions, page_dims)
#     ]

# #     pred = torch.Tensor(abs_coords[0])
#     # return (abs_coords,page_dims,regions)
#     return abs_coords


def rescaled_bboxes_from_cropped(img_cropped,img_source,top_left):
    left = top_left[0]
    top = top_left[1]
    
    img_source = cv2.cvtColor(cv2.imread(img_source),cv2.COLOR_BGR2RGB)
    target_h = img_source.shape[0]
    target_w = img_source.shape[1]
    
    img_cropp=[]
    img_cropp.append(img_cropped)
   
    result = predictor(img_cropp)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    # print(page_dims)
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    # abs_coords = [
    # [[int(round(word['geometry'][0][0] * target_w)), 
    #   int(round(word['geometry'][0][1] * target_h)), 
    #   int(round(word['geometry'][1][0] * target_w)), 
    #   int(round(word['geometry'][1][1] * target_h))] for word in words]
    # for words, dims in zip(regions, page_dims)
    # ]
    abs_coords = [
    [[int(round(word['geometry'][0][0] * dims[1]))+left, 
      int(round(word['geometry'][0][1] * dims[0]))+top, 
      int(round(word['geometry'][1][0] * dims[1]))+left, 
      int(round(word['geometry'][1][1] * dims[0]))+top] for word in words]
    for words, dims in zip(regions, page_dims)
    ]

#     pred = torch.Tensor(abs_coords[0])
    # return (abs_coords,page_dims,regions)
    return abs_coords

def visualize_preds_dir(img_dir):
    preds = doctr_predictions(img_dir)
    img = cv2.cvtColor(cv2.imread(img_dir),cv2.COLOR_BGR2RGB)
    for w in preds[0]:
        cv2.rectangle(img,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)
    # plt.imshow(img)
    cv2.imwrite('/home2/sreevatsa/output_test_doctrv2_{}_{}.png'.format(d,formatted_time), img)

def visualized_rescaled_bboxes_from_cropped(img_cropped,img_source,top_left):
    preds = rescaled_bboxes_from_cropped(img_cropped,img_source,top_left)
    img = cv2.cvtColor(cv2.imread(img_source),cv2.COLOR_BGR2RGB)
    for w in preds[0]:
        cv2.rectangle(img,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)
    # plt.imshow(img)
    # cv2.imwrite('/home2/sreevatsa/afterfixoutput_test_doctrv2_{}_{}.png'.format(d,formatted_time), cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def save_cropped(img_dir):
    img = cv2.cvtColor(cv2.imread(img_dir),cv2.COLOR_BGR2RGB)
    org_TL = (0,0)
    org_BR = (img.shape[1],img.shape[0])

    preds = doctr_predictions(img_dir)
    
    top1=[]
    left1=[]
    bottom1=[]
    right1 = []

    for i in preds[0]:
        left1.append(i[0])
        top1.append(i[1])
        right1.append(i[2])
        bottom1.append(i[3])

    l = min(left1)
    r = max(right1)
    t = min(top1)
    b = max(bottom1)
    # print(l,r,t,b)

    top_left = (l-20,t-20)
    bottom_right = (r+20,b+20)

    # print('cropped')
    # print(top_left, bottom_right)

    difference_TL = (top_left[0]-org_TL[0],top_left[1]-org_TL[1])
    difference_BR = (abs(bottom_right[0]-org_BR[0]),abs(bottom_right[1]-org_BR[1]))
    # print(difference_TL,difference_BR) 
    x1,y1 = top_left
    x2,y2 = bottom_right

    cv2.rectangle(img,top_left, bottom_right,(0,255,255),2)
    # plt.imshow(img1)
    imgg1 = img[y1:y2, x1:x2]
    # plt.imshow(imgg1)
    # cv2.imwrite('/home2/sreevatsa/cropped_image.png',imgg1) #no need of saving cropped image

    return top_left,imgg1

