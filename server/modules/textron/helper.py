import os
import shutil
import time
import uuid
from collections import OrderedDict
from os.path import join
from subprocess import check_output, run
from tempfile import TemporaryDirectory
from typing import List, Tuple

import torch
from fastapi import UploadFile

from ..core.config import *
from .model import *
from .src.config import *
from .main import textron_main

# TODO: remove this line and try to set the env from the docker-compose file.

def logtime(t: float, msg:  str) -> None:
	print(f'[{int(time.time() - t)}s]\t {msg}')

t = time.time()



def save_uploaded_image(image: UploadFile) -> str:

        t = time.time()
        print('removing all the previous uploaded files from the image folder')
        os.system(f'rm -rf {IMAGE_FOLDER}/*')
        os.system(f'rm -rf {os.path.join(PRED_IMAGES_FOLDER,"*")}')
        os.system(f'rm -rf {os.path.join(PRED_CAGE_FOLDER,"*")}')
        os.system(f'rm -rf {os.path.join(PRED_TXT_FOLDER,"*")}')
        location = join(IMAGE_FOLDER, '{}.{}'.format(
		str(uuid.uuid4()),
		image.filename.strip().split('.')[-1]
	))
        with open(location, 'wb+') as f:
                shutil.copyfileobj(image.file, f)
        logtime(t, 'Time took to save one image')
        return location

def process_images_textron(folder_path: str) -> List[LayoutImageResponse]:
    t = time.time()
    logtime(t, 'Time taken to load the textron model')
    print('running python TEXTRON MAIN FILE')
    run(["python",'-m', TEXTRON_MAIN_FILE])
    print('SUCCESSFULLY RAN THE TEXTRON MAIN FILE')
    # a=textron_main()
    t = time.time()
    logtime(t, 'Time taken to perform textron inference')
    img_files=os.listdir(folder_path)
    ret=[]
    for img_file in img_files:
                img_name=img_file.split('.')[0]
                with open(os.path.join(PRED_TXT_FOLDER,img_name)+'.txt','r',encoding='utf8') as f:
                        bboxes=f.read().split('\n')
                regions = []
                # bboxes=bboxes.split('\n')
                for bbox in bboxes:
                        if len(bbox.split())==6:
                                # print(bbox.split())
                                text,_,x,y,w,h=bbox.split()
                                # print([text,x,y,w,h])
                                regions.append(
                                       Region.from_bounding_box(
                                       BoundingBox(x=int(x),y=int(y),w=int(w),h=int(h)),
                                                                  label=text,
                                        )
								)
                # print('page done :',img_name)
                ret.append(
                        LayoutImageResponse(
                                image_name=img_name,
                                regions=regions.copy()
                        )
                )
    return ret

def textron_visulaize(img_file: str):
    t = time.time()
    print('running python TEXTRON MAIN FILE')
    run(["python",'-m', TEXTRON_MAIN_FILE])
    print('SUCCESSFULLY RAN THE TEXTRON MAIN FILE')
    # a=textron_main()
    t = time.time()