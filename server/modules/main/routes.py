import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import (process_image, process_image_craft,
                     process_image_worddetector, process_multiple_image_craft,
                     process_multiple_image_doctr,
                     process_multiple_image_doctr_v2, process_multiple_pages_ReadingOrderGenerator,
                     process_multiple_image_worddetector, save_uploaded_image)
from .models import LayoutImageResponse, ModelChoice
from .post_helper import process_dilate, process_multiple_dilate

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)


@router.post('/', response_model=List[LayoutImageResponse])
async def doctr_layout_parser(
	folder_path: str = Depends(save_uploaded_images),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
):
	"""
	API endpoint for calling the layout parser
	"""
	print(model.value)
	if model == ModelChoice.craft:
		ret = process_multiple_image_craft(folder_path)
	elif model == ModelChoice.worddetector:
		ret = process_multiple_image_worddetector(folder_path)
	elif model == ModelChoice.doctr:
		ret = process_multiple_image_doctr(folder_path)
	elif model == ModelChoice.v2_doctr:
		ret = process_multiple_image_doctr_v2(folder_path)
	elif model == ModelChoice.docTRv2_readingOrder:
		ret = process_multiple_pages_ReadingOrderGenerator(folder_path)
	if dilate:
		ret = process_multiple_dilate(ret)
	return ret


@router.post('/visualize')
async def layout_parser_swagger_only_demo(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
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
	elif model == ModelChoice.worddetector:
		regions = process_image_worddetector(image_path)
	else:
		regions = process_image(image_path, model.value)
	if dilate:
		regions = process_dilate(regions, image_path)
	save_location = '/home/layout/layout-parser/images/{}.jpg'.format(
		str(uuid.uuid4())
	)
	# TODO: all the lines after this can be transfered to the helper.py file
	bboxes = [i.bounding_box for i in regions]
	bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
	img = cv2.imread(image_path)
	count = 1
	for i in bboxes:
		img = cv2.rectangle(img, i[0], i[1], (0,0,255), 3)
		img = cv2.putText(
			img,
			str(count),
			(i[0][0]-5, i[0][1]-5),
			cv2.FONT_HERSHEY_COMPLEX,
			1,
			(0,0,255),
			1,
			cv2.LINE_AA
		)
		count += 1
	cv2.imwrite(save_location, img)
	return FileResponse(save_location)
