from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_image, save_uploaded_images
from .helper import process_textron_output, textron_visualize
from .models import LayoutImageResponse

router = APIRouter(
    prefix='/textron',
)

@router.post('', response_model=list[LayoutImageResponse])
async def perform_text_detection_using_textron(
    folder_path: str = Depends(save_uploaded_images),
) -> list[LayoutImageResponse]:
    """
    **NOT OPERATIONAL**
    API endpoint for calling the IITB Textron model
    """
    return process_textron_output(folder_path)


@router.post('/visualize')
async def visualize_text_detection_using_textron(
    image_path: str = Depends(save_uploaded_image),
):
    """
    **NOT OPERATIONAL**
    This endpoint is only used to demonstration purposes.
    this endpoint returns/displays the input image with the
    bounding boxes clearly marked in blue rectangles.

    PS: This endpoint is not to be called from outside of swagger
    """
    return FileResponse(textron_visualize(image_path))