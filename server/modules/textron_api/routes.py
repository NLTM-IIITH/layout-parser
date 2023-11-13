import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from subprocess import run

from .dependencies import save_uploaded_images
from .helper import save_uploaded_image,process_images_textron,textron_visulaize
from .model import LayoutImageResponse, ModelChoice
from .src.config import *
from ..core.config import *

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)


@router.post('/textron', response_model=List[LayoutImageResponse])
async def textron(
	folder_path: str = Depends(save_uploaded_images),
	# model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
):
    return process_images_textron(folder_path)

@router.post('/textron_visualization', response_model=List[LayoutImageResponse])
async def process_images_textron_visualization(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
):
    image_path = save_uploaded_image(image)
    img_name=image_path.split('/')[-1].split('.')[0]
    print(image_path)
    # regions = process_images_textron(image_path)
    print('RUNNING TEXTRON')
    a=textron_visulaize(image_path)
    save_location = PRED_IMAGES_FOLDER+'/{}_pred.jpg'.format(
		img_name
	)
    print('file_name SAVED :',save_location)
    return FileResponse(save_location)
