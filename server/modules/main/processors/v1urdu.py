import os
import time
from os.path import basename, join
from subprocess import check_output
from typing import Any

import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

def find_peaks_valley(hpp):
    line_index = []
    i = 0
    prev_i = -1
    while(i<len(hpp)-1):
        print("i==>",i)
        index1 = i
        flag1 = 0
        flag2 = 0
        for j in range (i, len(hpp)-1, 1):
            if (hpp[j] != 0):
                index1 = j-1
                line_index.append(index1)
                flag1 = 1
                break
        for j in range (index1+1, len(hpp)-1, 1):
            if (hpp[j] == 0 and flag1 ==1):
                index2 = j
                line_index.append(index2)
                flag2 = 1
                break
        if (flag1 ==1 and flag2==1):
            i = index2	
        if (flag1 == 0 and flag2 ==0):
            break
        if i == prev_i:
            break
        prev_i = i
    return line_index		

def binarize_image_sauvola(image):
    threshold = threshold_sauvola(image)
    image1 = image < threshold
    return image1 

class V1UrduProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        """
        given the path of the image, this function returns a list
        of bounding boxes of all the word detected regions.

        @returns list of BoundingBox class
        """
        ret = []
        for filename in os.listdir(folder_path):
            # image_path = folder_path + filename
            image_path = join(folder_path, filename)
            image = cv2.imread(image_path) # type: ignore
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # type: ignore
            _, image_w = image_gray.shape
            image_thres = binarize_image_sauvola(image_gray)
            print('binarize image completed')
            hpp = np.sum(image_thres, axis=1)
            print('horizontal projections completed')

            #finding vally and peak of histigram
            line_index = find_peaks_valley(hpp)
            print('found the peaks and valleys')

            regions = []
            line = 1
            for i in range(0,len(line_index)-1,2):
                y1 = int(line_index[i])
                y2 = int(line_index[i+1])
                x1 = 0
                x2 = image_w
                # color = (0,0,255)
                # thickness = 2 
                if (y2-y1>10):
                    regions.append(Region(
                        bounding_box=BoundingBox(
                            x=x1,
                            y=y1,
                            w=x2-x1,
                            h=y2-y1
                        ),
                        line=line,
                    ))
                    line += 1
            ret.append(LayoutImageResponse(
                image_name=basename(image_path),
                regions=regions.copy()
            ))
        return ret

register(ModelChoice.v1_urdu)(V1UrduProcessor())