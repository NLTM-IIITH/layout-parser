import base64
import os
import shutil
from os.path import basename, join, splitext
from subprocess import run

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import FileResponse

from server.config import settings
from server.modules.main.dependencies import save_uploaded_image

from .models import PreProcessorBinarizeResponse

router = APIRouter(
    prefix='/layout/preprocess',
    tags=['Preprocess'],
)


@router.post('/binarize', response_model=PreProcessorBinarizeResponse)
async def binarize_image(
    image_path: str = Depends(save_uploaded_image),
):
    """
    Returns the binarized image
    """
    run([
        'docker',
        'run', '-it', '--rm',
        '--gpus', 'all',
        '-v', f'{settings.image_folder}:/data',
        'preprocess:binarize',
        'python', 'infer.py',
    ], check=True)
    images = [i for i in os.listdir(settings.image_folder) if '_binarized' in i]
    if images:
        return FileResponse(
            join(settings.image_folder, images[0]),
        )
    else:
        raise FileNotFoundError("Binarized image not found. Please check the preprocessing step.")

@router.post('/binarize/docentr')
async def binarize_image_docentr(
    image_path: str = Depends(save_uploaded_image),
):
    """
    Returns the binarized image
    """
    run([
        'docker',
        'run', '-it', '--rm',
        '--gpus', 'all',
        '-v', f'{settings.image_folder}:/data',
        'preprocess-binarize:docentr',
        'python', 'infer.py',
    ], check=True)
    images = [i for i in os.listdir(settings.image_folder) if '_binarized' in i]
    if images:
        return FileResponse(
            join(settings.image_folder, images[0]),
        )
    else:
        raise FileNotFoundError("Binarized image not found. Please check the preprocessing step.")



@router.post('/binarize/bulk')
async def binarize_image_bulk(
    images: list[UploadFile]
):
    """
    Returns the binarized image
    """
    os.system(f'rm -rf {settings.image_folder}/*')
    print(f'Saving {len(images)} to location: {settings.image_folder}')
    out_images = []
    for idx, image in enumerate(images):
        fname = str(idx) + splitext(image.filename or 'image.jpg')[-1]
        location = join(settings.image_folder, fname)
        with open(location, 'wb') as f:
            shutil.copyfileobj(image.file, f)
        out_images.append(basename(location))
    run([
        'docker',
        'run', '-it', '--rm',
        '--gpus', 'all',
        '-v', f'{settings.image_folder}:/data',
        'preprocess:binarize',
        'python', 'infer.py',
    ], check=True)
    bimages = [i for i in os.listdir(settings.image_folder) if '_binarized' in i]
    bimages.sort(key=lambda x: int(x.split('_')[0]))
    assert len(bimages) == len(out_images), "Mismatch in number of binarized images and input images."
    bimages = [join(settings.image_folder, i) for i in bimages]
    bimages = list(map(convert_to_base64, bimages))
    return {
        'binarized_images': bimages,
        'message': 'Binarization completed successfully.'
    }

def convert_to_base64(image_path: str) -> str:
    """
    Convert an image to a base64 encoded string.
    """
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode()