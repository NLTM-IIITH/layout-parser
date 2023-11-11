import os
import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import (save_uploaded_image, process_multiple_image_fasterrcnn, process_image_fasterrcnn)
from .models import LayoutImageResponse, ModelChoice
from ..core.config import IMAGE_FOLDER


router = APIRouter(
	prefix='/layout',
	tags=['table'],
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
	ret = process_image_fasterrcnn(image_path)
	# Use os.path.join to create the complete save location
	save_location = image_path
	img = cv2.imread(image_path)
	count = 1


	for region in ret.regions:
		bounding_box = region.bounding_box
		cellrows = region.cellrows

		# Draw bounding box
		img = cv2.rectangle(img, (bounding_box.x, bounding_box.y), (bounding_box.x + bounding_box.w, bounding_box.y + bounding_box.h), (0, 0, 255), 3)
		img = cv2.putText(
			img,
			str(count),
			(bounding_box.x - 5, bounding_box.y - 5),
			cv2.FONT_HERSHEY_COMPLEX,
			1,
			(0, 0, 255),
			1,
			cv2.LINE_AA)
		count += 1

		if cellrows:
			for row, row_bboxes in cellrows.items():
				for cell_bbox in row_bboxes:
					# Draw cell bounding boxes (if available)
					img = cv2.rectangle(img, (cell_bbox.x, cell_bbox.y), (cell_bbox.x + cell_bbox.w, cell_bbox.y + cell_bbox.h), (0, 255, 0), 1)

	cv2.imwrite(save_location, img)
	return FileResponse(save_location)