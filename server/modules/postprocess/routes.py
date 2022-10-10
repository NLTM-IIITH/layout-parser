from subprocess import call
from typing import List

from fastapi import APIRouter

from .helper import process_images, process_ocr_output
from .models import SIRequest, SIResponse

router = APIRouter(
	prefix='/layout/postprocess',
	tags=['Postprocess'],
)


@router.post(
	'/script',
	response_model=List[SIResponse],
	response_model_exclude_none=True
)
def identify_script(si_request: SIRequest) -> List[SIResponse]:
	"""
	This is an endpoint for identifying the script of the word images.
	this model was contributed by **Punjab university (@Ankur)** on 07-10-2022
	The endpoint takes a list of images in base64 format and outputs the
	identified script for each image in the same order.

	Currently 8 recognized languages are [**hindi, telugu, tamil, gujarati,
	punjabi, urdu, bengali, english**]
	"""
	path = process_images(si_request.images)
	call(f'./script_iden_v1.sh', shell=True)
	return process_ocr_output()
