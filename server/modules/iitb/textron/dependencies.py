import os
import shutil
import uuid
from os.path import join
from typing import List

from fastapi import UploadFile

from server.modules.core.config import IMAGE_FOLDER


def save_uploaded_images(images: List[UploadFile]) -> str:
    print('removing all the previous uploaded files from the image folder')
    os.system(f'rm -rf {IMAGE_FOLDER}/*')
    print(f'Saving {len(images)} to location: {IMAGE_FOLDER}')
    for image in images:
        location = join(IMAGE_FOLDER, f'{image.filename.strip(" .")}')
        with open(location, 'wb') as f:
            shutil.copyfileobj(image.file, f)
    return IMAGE_FOLDER



def save_uploaded_image(image: UploadFile) -> str:
    """
    function to save the uploaded image to the disk

    @returns the absolute location of the saved image
    """
    print('removing all the previous uploaded files from the image folder')
    os.system(f'rm -rf {IMAGE_FOLDER}/*')
    location = join(IMAGE_FOLDER, '{}.{}'.format(
        str(uuid.uuid4()),
        image.filename.strip().split('.')[-1]
    ))
    with open(location, 'wb+') as f:
        shutil.copyfileobj(image.file, f)
    return location