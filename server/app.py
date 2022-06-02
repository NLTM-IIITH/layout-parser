import uuid

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from .helper import process_image, process_image_craft, save_uploaded_image
from .models import *

app = FastAPI(
	title='Layout Parser API',
	description='',
	docs_url='/',
)


@app.post('/layout', tags=['Main'], response_model=LayoutResponse)
async def doctr_layout_parser(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.doctr)
):
	"""
	API endpoint for ***doctr*** version of the layout parser
	"""
	image_path = save_uploaded_image(image)
	if model == ModelChoice.craft:
		regions = process_image_craft(image_path)
	else:
		regions = process_image(image_path)
	return LayoutResponse(regions=regions)


@app.post('/preprocess/binarize', tags=['Pre Process'], response_model=PreProcessorBinarizeResponse)
async def binarize_image(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorBinarizeResponse(images=regions)


@app.post('/preprocess/grayscale', tags=['Pre Process'], response_model=PreProcessorGrayScaleResponse)
async def grayscale_image(images: List[bytes] = File(...)):
	"""
	Returns the Grayscale image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorGrayScaleResponse(images=regions)


@app.post('/preprocess/color', tags=['Pre Process'], response_model=PreProcessorColorResponse)
async def Get_Image_Colors(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorColorResponse(images=regions)


@app.post('/preprocess/font', tags=['Pre Process'], response_model=PreProcessorFontResponse)
async def Get_Font_Properties_in_the_Image(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorFontResponse(images=regions)


@app.post('/preprocess/properties', tags=['Pre Process'], response_model=PreProcessorPropertiesResponse)
async def Get_Image_Properties(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorPropertiesResponse(images=regions)


@app.post('/layout/visualize', tags=['Visualization'])
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
	save_location = '/home/krishna/layout-parser/images/{}.jpg'.format(
		str(uuid.uuid4())
	)
	# TODO: all the lines after this can be transfered to the helper.py file
	bboxes = [i.bounding_box for i in regions]
	bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
	img = cv2.imread(image_path)
	for i in bboxes:
		img = cv2.rectangle(img, i[0], i[1], (0,0,255), 1)
	cv2.imwrite(save_location, img)
	return FileResponse(save_location)
