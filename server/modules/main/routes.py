import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import (process_image, process_image_craft,
                     process_multiple_image_craft,
                     process_multiple_image_doctr, save_uploaded_image)
from .models import LayoutImageResponse, ModelChoice

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)


@router.post('/', response_model=List[LayoutImageResponse])
async def doctr_layout_parser(
	folder_path: str = Depends(save_uploaded_images),
	model: ModelChoice = Form(ModelChoice.doctr)
):
	"""
	API endpoint for calling the layout parser
	"""
	print(model.value)
	if model == ModelChoice.craft:
		return process_multiple_image_craft(folder_path)
	elif model == ModelChoice.doctr:
		return process_multiple_image_doctr(folder_path)


@router.post('/visualize')
async def layout_parser_swagger_only_demo(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.doctr)
):
	"""
	This endpoint is only used to demonstration purposes.
	this endpoint returns/displays the input image with the
	bounding boxes clearly marked in blue rectangles.

	PS: This endpoint is not to be called from outside of swagger
	"""
	image_path = save_uploaded_image(image)
	if model == ModelChoice.craft:
		regions = process_image_craft(image_path)
	else:
		regions = process_image(image_path)
	save_location = '/home/layout/layout-parser/images/{}.jpg'.format(
		str(uuid.uuid4())
	)
	# TODO: all the lines after this can be transfered to the helper.py file
	bboxes = [i.bounding_box for i in regions]
	bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
	img = cv2.imread(image_path)
	for i in bboxes:
		img = cv2.rectangle(img, i[0], i[1], (0,0,255), 3)
	cv2.imwrite(save_location, img)
	return FileResponse(save_location)
