import uuid
from os.path import dirname

import cv2
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_image, save_uploaded_images
from .factory import PROCESSOR_REGISTORY
from .models import LayoutImageResponse, ModelChoice

router = APIRouter(
    prefix='/layout',
    tags=['Main'],
)


@router.get('/ping')
async def ping():
    return 'pong'


@router.post('/', response_model=list[LayoutImageResponse])
async def new_main_layout_parser(
    folder_path: str = Depends(save_uploaded_images),
    model: ModelChoice = Form(ModelChoice.doctr),
    language: str = Form('english'),
    registry: dict = Depends(lambda: PROCESSOR_REGISTORY)
) -> list[LayoutImageResponse]:
    try:
        processor = registry[model]
    except KeyError:
        raise HTTPException(400, f'Model {model.value} not supported')
    
    result = await processor(
        folder_path,
        language=language
    )
    return result

@router.post('/visualize')
async def new_main_layout_parser_visualize(
    image_path: str = Depends(save_uploaded_image),
    model: ModelChoice = Form(ModelChoice.doctr),
    language: str = Form('english'),
    order: bool = Form(True),
    line: bool = Form(False),
    center_point: bool = Form(True),
    font_size: float = Form(1.0, ge=0.1, le=1.0),
    registry: dict = Depends(lambda: PROCESSOR_REGISTORY)
) -> FileResponse:
    try:
        processor = registry[model]
    except KeyError:
        raise HTTPException(400, f'Model {model.value} not supported')
    
    folder_path = dirname(image_path)
    result = await processor(
        folder_path,
        language=language
    )
    regions = result[0].regions
    save_location = '/home/layout/layout-parser/images/{}.jpg'.format(
        str(uuid.uuid4())
    )
    # TODO: all the lines after this can be transfered to the helper.py file
    bboxes = [i.bounding_box for i in regions]
    orders = [i.order for i in regions]
    lines = [i.line for i in regions]
    bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h), j, k) for i,j,k  in zip(bboxes, orders, lines)]
    img = cv2.imread(image_path) # type: ignore
    for i in bboxes:
        overlay = img.copy()
        cv2.rectangle( # type: ignore
            overlay,
            i[0], i[1],
            (255, 0, 0) if i[2] == -1 else (0,255,0), -1
        )
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img) # type: ignore
        if center_point:
            cv2.circle( # type: ignore
                img,
                (
                    (i[0][0] + i[1][0]) // 2,
                    (i[0][1] + i[1][1]) // 2,
                ),
                3,
                (0, 0, 255),
                -1
            )
        if order:
            cv2.putText( # type: ignore
                img, 
                str(i[2]),
                (i[0][0]-5, i[0][1]-5),
                cv2.FONT_HERSHEY_COMPLEX, # type: ignore
                font_size,
                (0,0,255),
                1,
                cv2.LINE_AA # type: ignore
            )
        if line:
            cv2.putText( # type: ignore
                img, 
                str(i[3]),
                (i[0][0]-5, i[0][1]-5),
                cv2.FONT_HERSHEY_COMPLEX, # type: ignore
                font_size,
                (255,0,0),
                1,
                cv2.LINE_AA # type: ignore
            )
            
    cv2.imwrite(save_location, img) # type: ignore
    return FileResponse(save_location)
