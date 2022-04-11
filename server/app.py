import uuid

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from .helper import process_image, save_uploaded_image, sort_regions
from .models import *

app = FastAPI(
	title='Layout Parser API',
	description='',
	docs_url='/',
)


@app.post('/layout', tags=['Layout Parser'], response_model=LayoutResponse)
async def doctr_layout_parser(image: UploadFile = File(...)):
	"""
	API endpoint for ***doctr*** version of the layout parser
	"""
	image_path = save_uploaded_image(image)
	regions = process_image(image_path)
	regions = [Region.from_bounding_box(i) for i in regions]
	regions = sort_regions(regions)
	return LayoutResponse(regions=regions)


@app.post('/layout/demo', tags=['Demo'])
async def layout_parser_swagger_only_demo(
	image: UploadFile = File(...),
):
	"""
	This endpoint is only used to demonstration purposes.
	this endpoint returns/displays the input image with the
	bounding boxes clearly marked in blue rectangles.

	PS: This endpoint is not to be called from outside of swagger
	"""
	image_path = save_uploaded_image(image)
	regions = process_image(image_path)
	save_location = '/home/krishna/layout-parser/images/{}.jpg'.format(
		str(uuid.uuid4())
	)
	# TODO: all the lines after this can be transfered to the helper.py file
	regions = [Region.from_bounding_box(i) for i in regions]
	regions = sort_regions(regions)
	bboxes = [i.bounding_box for i in regions]
	bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
	img = cv2.imread(image_path)
	for i in bboxes:
		img = cv2.rectangle(img, i[0], i[1], (0,0,255), 1)
	cv2.imwrite(save_location, img)
	return FileResponse(save_location)
