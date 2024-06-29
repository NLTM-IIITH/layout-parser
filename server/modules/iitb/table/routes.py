import io
import json
import os
from subprocess import call
from tempfile import TemporaryDirectory
from typing import List

from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from .helper import save_uploaded_images

router = APIRouter(
    prefix='/table',
)

@router.post('', response_model=None)
async def table_layout_parser(
    images: List[UploadFile],
):
    """
    API endpoint for calling the layout parser
    """
    temp = TemporaryDirectory()

    save_uploaded_images(images, temp.name)
    
    print("Calling docker")
    call([
        "docker",
        "run",
        "--rm",
        "-v", f"{temp.name}:/model/data",
        "layout:iitb-table"]
    )
    print("Done docker")

    # Print the list of files
    print("Files in temporary directory:", os.listdir(temp.name))
    
    with open(os.path.join(temp.name, "out.json")) as f:
        out = json.load(f)	
    
    temp.cleanup()
    return out


@router.post('/visualize')
async def layout_parser_swagger_only_demo_table(
    images: List[UploadFile],
):
    """
    This endpoint is only used to demonstration purposes.
    this endpoint returns/displays the input image with the
    bounding boxes clearly marked in blue rectangles.

    PS: This endpoint is not to be called from outside of swagger
    """
    temp = TemporaryDirectory()
    folder = temp.name
    # folder = '/home/layout/temp'

    save_uploaded_images(images, folder)

    print("Calling docker")
    call([
        "docker",
        "run",
        "--rm",
        "-v", f"{folder}:/model/data",
        "layout:iitb-table"]
    )
    print("Done docker")

    # Print the list of files
    print("Files in temporary directory:", os.listdir(folder))
    image_path_with_boxes = os.path.join(folder, "boxes.jpg")

    with open(image_path_with_boxes, mode="rb") as img_file:
        return StreamingResponse(io.BytesIO(img_file.read()), media_type="image/jpeg")