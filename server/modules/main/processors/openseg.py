import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from os.path import basename, join
from typing import Any

import pytesseract

from server.config import settings

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

def process_single_tesseract(image_info):
    image_path, language = image_info
    print(f'Processing tesseract for file: {basename(image_path)}; Time taken: ', end='')
    t = time.time()
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
    print(f'{round(time.time() - t, 2)}s')
    return LayoutImageResponse(
        image_name=basename(image_path),
        regions=regions
    )

class OpensegProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        """
        given the path of the image, this function returns a list
        of bounding boxes of all the word detected regions.

        @returns list of BoundingBox class
        """
        language = kwargs.get('language', 'english')
        ret = []
        image_paths = await asyncio.to_thread(os.listdir, folder_path)
        image_paths = [join(folder_path, i) for i in image_paths]
        image_infos = [(i, language) for i in image_paths]
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=5) as executor:
            tasks = [
                loop.run_in_executor(executor, process_single_tesseract, info)
                for info in image_infos
            ]
            ret = await asyncio.gather(*tasks)
        return ret

register(ModelChoice.openseg)(OpensegProcessor())