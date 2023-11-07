from fastapi import APIRouter, File, UploadFile, Form

from ..main.helper import process_image, save_uploaded_image
from .models import *

from .helper import save_uploaded_images, process_image_font_attributes_doctr, process_image_font_attributes_tesseract

router = APIRouter(
	prefix='/layout/preprocess',
	tags=['Preprocess'],
)


@router.post('/binarize', response_model=PreProcessorBinarizeResponse)
async def binarize_image(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorBinarizeResponse(images=regions)


@router.post('/grayscale', response_model=PreProcessorGrayScaleResponse)
async def grayscale_image(images: List[bytes] = File(...)):
	"""
	Returns the Grayscale image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorGrayScaleResponse(images=regions)


@router.post('/color', response_model=PreProcessorColorResponse)
async def Get_Image_Colors(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorColorResponse(images=regions)


@router.post('/font', response_model=PreProcessorFontResponse)
async def Get_Font_Properties_in_the_Image(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorFontResponse(images=regions)


@router.post('/properties', response_model=PreProcessorPropertiesResponse)
async def Get_Image_Properties(images: List[bytes] = File(...)):
	"""
	Returns the binarized image
	"""
	image_path = save_uploaded_image(images)
	regions = process_image(image_path)
	return PreProcessorPropertiesResponse(images=regions)

@router.post('/font_v2',response_model=FontAttributesResponse)
async def get_font_properties_from_image(
	images: List[UploadFile],
	model: ModelChoice = Form(ModelChoice.doctr),
	modality: ModalityChoice = Form(ModalityChoice.word),
	k_size: int = Form(default=4),
	bold_threshold: float = Form(default=0.3)
	):
	"""
	This endpoint returns the font attributes of text in the image.
	"""

	image_path = save_uploaded_images(images)
	
	if model == ModelChoice.doctr:
		font_attribute_images = process_image_font_attributes_doctr(image_path,k_size=k_size,bold_threshold=bold_threshold)
	elif model == ModelChoice.tesseract:
		font_attribute_images = process_image_font_attributes_tesseract(image_path)

	return FontAttributesResponse(images=font_attribute_images)