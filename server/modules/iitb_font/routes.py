from fastapi import APIRouter, UploadFile, Form

from .models import *

from .helper import save_uploaded_images, process_image_font_attributes_doctr, process_image_font_attributes_tesseract

router = APIRouter(
	prefix='/layout/iitb-font',
	tags=['IITB Font'],
)

@router.post('/font',response_model=FontAttributesResponse)
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