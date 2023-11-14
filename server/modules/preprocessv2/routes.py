import os
import cv2
from fastapi import File, UploadFile, Form
from fastapi.responses import FileResponse

from .models import *

from .helper import save_uploaded_images,save_uploaded_image, process_image_font_attributes_doctr, process_image_font_attributes_tesseract

from ..preprocess.routes import router


@router.post('/font_v2',response_model=FontAttributesResponse)
async def get_font_properties_from_image(
	images: List[UploadFile],
	model: ModelChoice = Form(ModelChoice.doctr),
	k_size: int = Form(default=4),
	bold_threshold: float = Form(default=0.3)
	):
	"""
	This endpoint returns the font attributes of text from images.
	"""

	image_path = save_uploaded_images(images)
	
	if model == ModelChoice.doctr:
		font_attribute_images = process_image_font_attributes_doctr(image_path,k_size=k_size,bold_threshold=bold_threshold)
	elif model == ModelChoice.tesseract:
		font_attribute_images = process_image_font_attributes_tesseract(image_path,k_size=k_size,bold_threshold=bold_threshold)

	return FontAttributesResponse(images=font_attribute_images)

@router.post('/visualize_font')
async def font_attribute_visualize(
	images: UploadFile=File(...),
	model: ModelChoice = Form(ModelChoice.doctr),
	k_size: int = Form(default=4),
	bold_threshold: float = Form(default=0.3)
	):
	"""
	This endpoint returns the font attributes of text from images.
	"""

	image_path, image_folder = save_uploaded_image(images)
	
	if model == ModelChoice.doctr:
		font_attribute_images = process_image_font_attributes_doctr(image_folder,k_size=k_size,bold_threshold=bold_threshold)
	elif model == ModelChoice.tesseract:
		font_attribute_images = process_image_font_attributes_tesseract(image_folder,k_size=k_size,bold_threshold=bold_threshold)

	font_regions = font_attribute_images[0].font_regions
	img = cv2.imread(image_path)

	for font_region in font_regions:
		bbox = font_region.bounding_box
		coords = ((bbox.x,bbox.y),(bbox.x+bbox.w,bbox.y+bbox.h))
		color = (0,0,255) if font_region.fontDecoration=="regular" else (0,255,0)
		img = cv2.rectangle(img,coords[0],coords[1],color,3)
		if model == ModelChoice.tesseract:
			img = cv2.putText(
				img,
				str(font_region.fontSize),
				(coords[0][0]-5, coords[0][1]-5),
				cv2.FONT_HERSHEY_COMPLEX,
				1,
				color,
				1,
				cv2.LINE_AA
			)
	
	visualize_result = os.path.join(image_folder,"visualize.png")
	cv2.imwrite(visualize_result,img)
	return FileResponse(visualize_result)
