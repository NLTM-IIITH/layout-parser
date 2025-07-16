import os
import shutil
import uuid
from os.path import join
from typing import List

from fastapi import UploadFile

from server.config import settings


def save_uploaded_images(images: List[UploadFile]) -> str:
	print('removing all the previous uploaded files from the image folder')
	os.system(f'rm -rf {settings.image_folder}/*')
	print(f'Saving {len(images)} to location: {settings.image_folder}')
	for image in images:
		location = join(settings.image_folder, f'{image.filename.strip(" .")}')
		with open(location, 'wb') as f:
			shutil.copyfileobj(image.file, f)
	return settings.image_folder

def save_uploaded_image(image: UploadFile) -> str:
    """
    function to save the uploaded image to the disk

    @returns the absolute location of the saved image
    """
    print('removing all the previous uploaded files from the image folder')
    os.system(f'rm -rf {settings.image_folder}/*')
    location = join(settings.image_folder, '{}.{}'.format(
        str(uuid.uuid4()),
        image.filename.strip().split('.')[-1]
    ))
    with open(location, 'wb+') as f:
        shutil.copyfileobj(image.file, f)
    return location