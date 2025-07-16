from os.path import basename
from typing import Any

import pytesseract
from ultralytics import YOLO

from server.config import settings

from ..factory import register
from ..helper import merge_ajoy_with_openseg
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def process_single_tesseract(image_info):
    image_path, language = image_info
    print(f'Processing tesseract for file: {basename(image_path)}; Time taken: ', end='')
    lang = settings.tesseract_language.get(language, 'eng')
    if lang == 'hin':
        lang = 'pan+hin+eng'
    elif lang in ('tel', 'tal'):
        lang = 'tel+tal'
    elif lang in ('guj', 'ori', 'mal', 'kan'):
        lang = f'{lang}+tel+tal'
    elif lang == 'eng':
        lang = 'eng'
    else:
        lang = f'{lang}+hin'
    results = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT, lang=lang)

    regions = []
    order = 1
    for i in range(len(results['text'])):
        if int(results['conf'][i]) <= 0:
            # Skipping the region as confidence is too low
            continue
        x = results['left'][i]
        y = results['top'][i]
        w = results['width'][i]
        h = results['height'][i]
        if h < 10 or w < 3:
            continue
        regions.append(Region(
            bounding_box=BoundingBox(
                x=x,
                y=y,
                w=w,
                h=h
            ),
            line=results['line_num'][i] + 1,
            order=order,
        ))
        order += 1
    return LayoutImageResponse(
        image_name=basename(image_path),
        regions=regions
    )



class V04xxProcessor:

    def __init__(self, model: str = 'V-04.01') -> None:
        self.model = model
        return super().__init__()

    async def __call__(self, folder_path: str, **kwargs: Any):
        language = kwargs.get('language', 'english')
        model_iden = int(self.model.split('.')[-1])
        model_path = settings.v04xx_model_path / f'{model_iden}.pt'
        model = YOLO(model_path)
        results = model.predict(source=folder_path, max_det=4000)
        ret = []
        for result in results:
            bboxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist() # type: ignore
            bboxes = [tuple(i) for i in bboxes]
            confs = (result.boxes.conf.cpu().numpy().round(3)) # type: ignore
            regions = [Region.from_xyxy(v, line=i+1, conf=c, order=i+1) for i, (v, c) in enumerate(zip(bboxes, confs))]
            # openseg_regions = process_single_tesseract((result.path, language)).regions
            # regions = merge_ajoy_with_openseg(regions, openseg_regions)
            # regions = [Region(**i) for i in regions]
            response = LayoutImageResponse(
                image_name=basename(result.path),
                regions=regions.copy(),
            )
            ret.append(response)
        return ret

register(ModelChoice.v0401)(V04xxProcessor(model='V-04.01'))
register(ModelChoice.v0402)(V04xxProcessor(model='V-04.02'))
register(ModelChoice.v0403)(V04xxProcessor(model='V-04.03'))
register(ModelChoice.v0404)(V04xxProcessor(model='V-04.04'))
register(ModelChoice.v0405)(V04xxProcessor(model='V-04.05'))