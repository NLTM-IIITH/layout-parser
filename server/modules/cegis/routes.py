import uuid
from tempfile import TemporaryDirectory
from typing import List

import cv2
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from .helper import (get_all_images, perform_align, save_image,
                     save_template_coords, save_template_image)
from .models import LayoutIn, LayoutOut

router = APIRouter(
	prefix='/layout/cegis',
	tags=['CEGIS'],
)


@router.post('/', response_model=LayoutOut)
async def cegis_layout_parser(
	data: LayoutIn
) -> LayoutOut:
	"""
	API endpoint for ***cegis*** version of the layout parser
	"""
	tmp = TemporaryDirectory(prefix='cegis')
	tmp2 = TemporaryDirectory(prefix='cegis_output')
	image_path = save_image(data.image, tmp.name)
	template_image_path = save_template_image(data.template_image, tmp.name)
	template_coords_path = save_template_coords(data.template_coords, tmp.name)
	print(image_path, template_image_path, template_coords_path)
	perform_align(image_path, tmp2.name, template_image_path, template_coords_path)
	return LayoutOut(images=get_all_images(tmp2.name))
