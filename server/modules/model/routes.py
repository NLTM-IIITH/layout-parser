import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import (save_uploaded_image, process_multiple_image_fasterrcnn, process_image_fasterrcnn)
from .models import LayoutImageResponse, ModelChoice
from .post_helper import process_dilate, process_multiple_dilate

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)

@router.post('/table', response_model=List[LayoutImageResponse])
async def table_layout_parser(
	folder_path: str = Depends(save_uploaded_images),
	model: ModelChoice = Form(ModelChoice.fasterrcnn),
	dilate: bool = Form(False),
):
	"""
	API endpoint for calling the layout parser
	"""
	print(model.value)
	ret = process_multiple_image_fasterrcnn(folder_path)
	return ret

@router.post('/visualize/table')
async def layout_parser_swagger_only_demo_table(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.fasterrcnn),
	dilate: bool = Form(False),
):
	"""
	This endpoint is only used to demonstration purposes.
	this endpoint returns/displays the input image with the
	bounding boxes clearly marked in blue rectangles.

	PS: This endpoint is not to be called from outside of swagger
	"""
	image_path = save_uploaded_image(image)
	regions = process_image_fasterrcnn(image_path)
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