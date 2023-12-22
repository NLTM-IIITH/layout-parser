import os
import shutil
import json
from subprocess import call,check_output
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
from .models import *

# TODO: remove this line and try to set the env from the docker-compose file.

def logtime(t: float, msg:  str) -> None:
	print(f'[{int(time.time() - t)}s]\t {msg}')

t = time.time()



def process_textron_output(folder_path: str) -> List[LayoutImageResponse]:
    try:
        # call(f'./textron.sh', shell=True)
        print(os.listdir(IMAGE_FOLDER))
        try:
            check_output(['docker','run','--rm','--net','host','-v',f'{IMAGE_FOLDER}:/data','textron:1'])
        except:
            check_output(['sudo','docker','run','--rm','--net','host','-v',f'{IMAGE_FOLDER}:/data','textron:1'])
        a = open(IMAGE_FOLDER+'/out.json', 'r').read().strip()
        a = json.loads(a)
        ret=[]
        for page in a.keys():
            regions=[]
            for bbox in a[page]:
                regions.append(
                    Region.from_bounding_box(
                        BoundingBox(x=int(bbox['x']),y=int(bbox['y']),w=int(bbox['w']),h=int(bbox['h']),
                            label=bbox['label'],
                        )
					)
                )
            ret.append(
                LayoutImageResponse(
                  image_name=page,
                  regions=regions.copy()
                )
            )
        return ret     
                     
    except Exception as e:
        print(e)

def textron_visualize(image_path: str) -> List[Region]:
    try:
        check_output(['docker','run','--rm','--net','host','-v',f'{IMAGE_FOLDER}:/data','textron:1'])
    except:
        check_output(['sudo','docker','run','--rm','--net','host','-v',f'{IMAGE_FOLDER}:/data','textron:1'])
    a = open(IMAGE_FOLDER+'/out.json', 'r').read().strip()
    a = json.loads(a)
    for page in a.keys():
        regions=[]
        for bbox in a[page]:
            regions.append(
                Region.from_bounding_box(
                    BoundingBox(x=int(bbox['x']),y=int(bbox['y']),w=int(bbox['w']),h=int(bbox['h']),
                        label=bbox['label'],
                    )
                )
            )
        return regions