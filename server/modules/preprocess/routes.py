from fastapi import APIRouter, File

from ..main.helper import process_image, save_uploaded_image
from .models import *

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
