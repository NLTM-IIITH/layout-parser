import uuid
# from .croppadfix import *
from tempfile import TemporaryDirectory
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import (Reading_Order_Generator, cropPadFix, process_image,
                     process_image_craft, process_image_urdu_v1,
                     process_image_worddetector, process_multiple_image_craft,
                     process_multiple_image_doctr,
                     process_multiple_image_doctr_v2,
                     process_multiple_image_worddetector,
                     process_multiple_pages_ReadingOrderGenerator,
                     process_multiple_tesseract, process_multiple_urdu_v1,
                     save_uploaded_image, process_multiple_image_cropPadFix)
from .models import LayoutImageResponse, ModelChoice
from .post_helper import process_dilate, process_multiple_dilate
from .readingOrder import *
from .textualAttribute import *

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)


@router.post('/', response_model=List[LayoutImageResponse])
async def doctr_layout_parser(
	folder_path: str = Depends(save_uploaded_images),
	model: ModelChoice = Form(ModelChoice.doctr),
	language: str = Form('english'),
	dilate: bool = Form(False),
	left_right_percentages: int = Form(0),
	header_percentage: int = Form(0),
	footer_percentage: int = Form(0)	
	#TODO: option to add True/False selection for using yolo for layout detection
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
	elif model == ModelChoice.v2_docTR_readingOrder:
		ret = process_multiple_pages_ReadingOrderGenerator(folder_path, left_right_percentages, header_percentage, footer_percentage)
	elif model == ModelChoice.v1_urdu:
		ret = process_multiple_urdu_v1(folder_path)
	elif model == ModelChoice.tesseract:
		ret = process_multiple_tesseract(folder_path, language)
	elif model ==ModelChoice.v1_textAttb:
		# order is working
		tmp = TemporaryDirectory(prefix="misc")
		ret = process_multiple_pages_TextualAttribute(folder_path,tmp.name)
	elif model == ModelChoice.cropPadFix:
		ret = process_multiple_image_cropPadFix(folder_path)
	if dilate:
		ret = process_multiple_dilate(ret)
	return ret


@router.post('/visualize')
async def layout_parser_swagger_only_demo(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
	language: str = Form('english'),
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
	elif model == ModelChoice.v1_urdu:
		regions = process_image_urdu_v1(image_path)
	elif model == ModelChoice.tesseract:
		folder_path = os.path.dirname(image_path)
		regions = process_multiple_tesseract(folder_path, language)[0].regions
	else:
		regions = process_image(image_path, model.value)
	if dilate:
		regions = process_dilate(regions, image_path)
	# save_location = '/home/layout/layout-parser/images/{}.jpg'.format(
	# 	str(uuid.uuid4())
	# )
	os.makedirs('/ssd_scratch/sreevatsa/layout-parser/saved_images', exist_ok = True)
	save_location = '/ssd_scratch/sreevatsa/layout-parser/saved_images/{}.jpg'.format(str(uuid.uuid4()))
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


@router.post('/visualize/readingorder')
async def layout_parser_swagger_only_demo_Reading_Order(
	image: UploadFile = File(...),
	left_right_percentage: int = Form(
		0,
		ge=0,
		le=100,
		description='Left right margins in percent of the total page width'
	),
	header_percentage: int = Form(
		0,
		ge=0,
		le=100,
		description='Header margin in percent of the total page height from top'
	),
	footer_percentage: int = Form(
		0,
		ge=0,
		le=100,
		description='Footer margin in percent of the total page height from bottom'
	),
	para_only: bool = False,
	col_only:bool = False,
	use_yolo: bool = True
):
	"""
	This endpoint is only used to demonstration purposes.
	this endpoint returns/displays the input image with the
	bounding boxes clearly marked in blue rectangles.

	PS: This endpoint is not to be called from outside of swagger
	"""
	image_path = save_uploaded_image(image)
	save_location = '/ssd_scratch/sreevatsa/layout-parser/saved_images/{}.jpg'.format(str(uuid.uuid4()))
	if para_only is True and col_only is False:
		img = Reading_Order_Generator(image_path, left_right_percentage, header_percentage, footer_percentage, para_only,col_only)
		cv2.imwrite(save_location, img)
		return FileResponse(save_location)
	elif para_only is False and col_only is True:
		img = Reading_Order_Generator(image_path, left_right_percentage, header_percentage, footer_percentage, para_only,col_only)
		cv2.imwrite(save_location, img)
		return FileResponse(save_location)
	elif para_only is True and col_only is True:
		pass
	elif para_only is False and col_only is False:
		img,_ = Reading_Order_Generator(image_path, left_right_percentage, header_percentage, footer_percentage, para_only,col_only, use_yolo)
		cv2.imwrite(save_location,img)
		return FileResponse(save_location)


@router.post('/visualize/paragraph_order')
async def layout_parser_swagger_only_demo_Paragraph_Reading_Order(
	image: UploadFile = File(...),
	left_right_percentage: int = Form(
		0,
		ge=0,
		le=100,
		description='Left right margins in percent of the total page width'
	),
	header_percentage: int = Form(
		0,
		ge=0,
		le=100,
		description='Header margin in percent of the total page height from top'
	),
	footer_percentage: int = Form(
		0,
		ge=0,
		le=100,
		description='Footer margin in percent of the total page height from bottom'
	)	
):
	"""
	This endpoint is only used to demonstration purposes.
	this endpoint returns/displays the input image with the
	paragraph bounding boxes clearly marked in blue rectangles.

	PS: This endpoint is not to be called from outside of swagger
	"""
	image_path = save_uploaded_image(image)
	save_location = '/home/layout/layout-parser/images/{}.jpg'.format(str(uuid.uuid4()))
	para_only = True
	img = Reading_Order_Generator(image_path, left_right_percentage, header_percentage, footer_percentage, para_only)
	cv2.imwrite(save_location, img)
	return FileResponse(save_location)


#croppadfix
@router.post('/visualize/croppadfixForTextBooks')
async def layout_parser_swagger_only_demo_Crop_Pad_fix(
	image: UploadFile = File(...)	
):
	"""
	This endpoint is only used to demonstration purposes.
	this endpoint returns/displays the input image with the
	bounding boxes clearly marked in rectangles.

	PS: This endpoint is not to be called from outside of swagger
	"""
	image_path = save_uploaded_image(image)
	save_location = '/home/layout/layout-parser/images/{}.jpg'.format(str(uuid.uuid4()))
	
	img_cpf = cropPadFix(image_path)
	cv2.imwrite(save_location,img_cpf)
	return FileResponse(save_location)

@router.post('/visualize/textual_attribute')
async def layout_parser_swagger_only_demo_Textual_Attribute(
	image: UploadFile = File(...)
):
	"""
	<h2>Description : </h2>\n
	This API is used to classify the word based textual attributes like None, Bold, Italic or BoldItalic 	 based on the word level features (V1 version : December 8 2023). It currently runs well on Scanned Notices , Govt. Circulars and Lok Sabha documents.\n

	<h2>Input : </h2>\n
	You need to upload a single `.jpg` file document image. 

	<h2>Interpretation of Output : </h2>\n
	This endpoint is only used to demonstration purposes. \n
	This endpoint returns/displays the input image with the
	bounding boxes clearly marked in colored rectangles based on the textual attribute : \n
	1. For no attribute : Blue color
	2. For bold attribute: Red Color
	3. For italic attribute : Green Color
	4. For bold+italic attribute : Purple color

	PS: This endpoint is not to be called from outside of swagger
	"""

	image_path = save_uploaded_image(image) 
	# saving the uploaded image

	"""
	My working and model functions (don't edit)
	"""

	tmp = TemporaryDirectory(prefix="misc")
	output_image = Visualization(image_location=image_path).visualize(temp_file=tmp.name)
	"""
	End
	"""

	save_location = '/home/layout/layout-parser/images/{}.jpg'.format(str(uuid.uuid4()))
	cv2.imwrite(save_location,output_image)
	return FileResponse(save_location)
