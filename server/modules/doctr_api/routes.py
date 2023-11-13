import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import process_multiple_image_doctr_v2,save_uploaded_image
from .models import LayoutImageResponse, ModelChoice

router = APIRouter(
	prefix='/text-recognition',
	tags=['Main'],
)


@router.post('/', response_model=List[LayoutImageResponse])
async def doctr_layout_parser(
	folder_path: str = Depends(save_uploaded_images),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
):
    ret = process_multiple_image_doctr_v2(folder_path)
    return ret


@router.post('/visualization', response_model=List[LayoutImageResponse])
async def doctr_layout_parser_visualization(
	image: UploadFile = File(...),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
):
    image_path = save_uploaded_image(image)
    regions = process_multiple_image_doctr_v2(image_path)
    bboxes = [i.bounding_box for i in regions]
    bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
    img = cv2.imread(image_path)
    count = 1
    for i in bboxes:
        img = cv2.rectangle(img, i[0], i[1], (0,0,255), 3)
        img = cv2.putText(
            img,
            str(count),
            (i[0][0]-5, i[0][1]-5),
            cv2.FONT_HERSHEY_COMPLEX,
			1,
			(0,0,255),
			1,
			cv2.LINE_AA
		)
        count += 1
    save_location = '/home/layout/layout-parser/images/{}.jpg'.format(
		str(uuid.uuid4())
	)
    cv2.imwrite(save_location, img)
    return FileResponse(save_location)