import json
import os
import pickle
from os.path import join
from subprocess import check_output
from tempfile import TemporaryDirectory

import cv2
from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import Response

from .helper import save_uploaded_images
from .models import ModelChoice, TaskChoice

router = APIRouter(
    prefix='/textattrib',
)


@router.post('', response_model=None)
async def get_font_properties_from_image(
    images: list[UploadFile],
    model: ModelChoice = Form(ModelChoice.doctr),
    task: TaskChoice = Form(TaskChoice.attributes),
    k_size: int = Form(default=4),
    bold_threshold: float = Form(default=0.3)
):
    """
    This endpoint returns the font attributes of text from images.
    """
    temp = TemporaryDirectory()
    folder = temp.name
    image_path = save_uploaded_images(images, folder)
    
    config = {
        "model": "doctr" if model == ModelChoice.doctr else "tesseract",
        "k_size": k_size,
        "bold_threshold": bold_threshold
    }

    with open(join(image_path, "config"), "wb") as f:
        pickle.dump(config, f)

    print("Calling docker")
    check_output([
        'docker', 
        'run',
        '--rm',
        '-v', f'{folder}:/model/data',
        'layout:iitb-textattrib'
    ])
    print("Done docker")
    

    if task == TaskChoice.attributes:
        with open(join(folder, "out.json")) as f:
            out = json.load(f)	
        # response = FontAttributesResponse.model_validate(out)
        response = out
    
    else:
        result_images = [join(folder, "result", i) for i in os.listdir(join(folder, "result"))]
        img = cv2.imread(result_images[0])
        _, im_png = cv2.imencode(".png", img)
        response = Response(content=im_png.tobytes(), media_type="image/png")
    
    # temp.cleanup()
    return response