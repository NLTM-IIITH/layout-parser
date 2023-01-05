import os
from tempfile import TemporaryDirectory
from typing import Dict, List

from fastapi import APIRouter

from .helper import (get_all_images, perform_align, save_image,
                     save_template_coords, save_template_image)
from .models import LayoutIn, LayoutOut

router = APIRouter(
	prefix='/layout/cegis',
	tags=['CEGIS'],
)


@router.post('/', response_model=Dict[str, Dict[str, List[str]]])
async def cegis_layout_parser(
	data: LayoutIn
) -> LayoutOut:
	"""
	API endpoint for ***cegis*** version of the layout parser
	"""
	tmp = TemporaryDirectory(prefix='cegis')
	print(data)
	tmp2 = TemporaryDirectory(prefix='cegis_output')
	# image_path = save_image(data.image, tmp.name)
	# template_image_path = save_template_image(data.template_image, tmp.name)
	# template_coords_path = save_template_coords(data.template_coords, tmp.name)
	template_image_path, ts = save_template_image(data.template_image, '/home/layout/temp_image')
	image_path = save_image(data.image, '/home/layout/temp_image', ts)
	template_coords_path = save_template_coords(data.template_coords, '/home/layout/temp_image')
	print(image_path, template_image_path, template_coords_path)
	# perform_align(image_path, tmp2.name, template_image_path, template_coords_path)
	perform_align(image_path, '/home/layout/out', template_image_path, template_coords_path)
	# return get_all_images(tmp2.name)
	ret = get_all_images('/home/layout/out')
	os.system('rm -rf /home/layout/out/*')
	return ret
