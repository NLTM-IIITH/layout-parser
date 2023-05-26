import os
import shutil
from os.path import join
from tempfile import TemporaryDirectory
from typing import List

from fastapi import File, UploadFile

from ..core.config import IMAGE_FOLDER


def save_uploaded_images(
	images: List[UploadFile],
	model: str,
):
	if model != 'craft':
		print('removing all the previous uploaded files from the image folder')
		tmp = TemporaryDirectory(prefix='images')
		print(f'Saving {len(images)} to location: {tmp.name}')
		for image in images:
			location = join(tmp.name, f'{image.filename}')
			with open(location, 'wb') as f:
				shutil.copyfileobj(image.file, f)
		return tmp
	else:
		print('removing all the previous uploaded files from the image folder')
		os.system(f'rm -rf {IMAGE_FOLDER}/*')
		print(f'Saving {len(images)} to location: {IMAGE_FOLDER}')
		for image in images:
			location = join(IMAGE_FOLDER, f'{image.filename}')
			with open(location, 'wb') as f:
				shutil.copyfileobj(image.file, f)
		return IMAGE_FOLDER
