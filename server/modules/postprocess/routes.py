from subprocess import call
from tempfile import TemporaryDirectory

from fastapi import APIRouter

from .helper import process_images, process_layout_output
from .models import MIResponse, PostprocessRequest, SIResponse

router = APIRouter(
	prefix='/layout/postprocess',
	tags=['Postprocess'],
)

@router.post(
	'/language/scenetext',
	response_model=list[SIResponse],
	response_model_exclude_none=True,
)
def identify_language(si_request: PostprocessRequest) -> list[SIResponse]:
	"""
	This is the endpoint for classifying the language of the **REAL** Scenetext images.
	this model works for all the 14 language (13 Indian + english)
	"""
	tmp = TemporaryDirectory(prefix='st_language_classify')
	process_images(si_request.images, tmp.name)
	call(f'./lang_iden_v1.sh {tmp.name}', shell=True)
	return process_layout_output(tmp.name)


@router.post(
	'/script',
	response_model=list[SIResponse],
	response_model_exclude_none=True
)
def identify_script(si_request: PostprocessRequest) -> list[SIResponse]:
	"""
	This is an endpoint for identifying the script of the word images.
	this model was contributed by **Punjab university (@Ankur)** on 07-10-2022
	The endpoint takes a list of images in base64 format and outputs the
	identified script for each image in the same order.

	Currently 8 recognized languages are [**hindi, telugu, tamil, gujarati,
	punjabi, urdu, bengali, english**]
	"""
	tmp = TemporaryDirectory(prefix='st_language_classify')
	process_images(si_request.images, tmp.name)
	call(f'./script_iden_v1.sh {tmp.name}', shell=True)
	return process_layout_output(tmp.name)


@router.post(
	'/modality',
	response_model=list[MIResponse],
	response_model_exclude_none=True,
)
def identify_language(si_request: PostprocessRequest) -> list[MIResponse]:
	"""
	This is the endpoint for classifying the modality of the images.
	this model works for all the 14 language (13 Indian + english) and
	outputs among 3 classes ["**printed**", "**handwritten**", "**scenetext**"]
	"""
	tmp = TemporaryDirectory(prefix='modality_classify')
	process_images(si_request.images, tmp.name)
	call(f'./modality_iden_v1.sh {tmp.name}', shell=True)
	return process_layout_output(tmp.name)