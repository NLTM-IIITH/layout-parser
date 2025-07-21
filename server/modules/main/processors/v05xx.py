import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from os.path import basename, join
from typing import Any

import pytesseract
from ultralytics import YOLO

from server.config import settings

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region
from .craft import CRAFTProcessor
from .merge_codes.merge_ajoy_openseg import merge_ajoy_openseg
from .merge_codes.merge_ajoy_openseg_craft import merge_all_regions
from .merge_codes.merge_ajoy_openseg_craft_v3 import merge_3_new


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



class V05xxProcessor:

    def __init__(self, model: str = 'V-05.02') -> None:
        self.model = model
        return super().__init__()

    async def ajoy_infer(self, folder_path: str) -> list[LayoutImageResponse]:
        model_path = settings.v04xx_model_path / '2.pt'
        model = YOLO(model_path)
        results = model.predict(source=folder_path, max_det=4000)
        ret = []
        for result in results:
            bboxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist() # type: ignore
            bboxes = [tuple(i) for i in bboxes]
            confs = (result.boxes.conf.cpu().numpy().round(3)) # type: ignore
            regions = [Region.from_xyxy(v, line=i+1, conf=c, order=i+1) for i, (v, c) in enumerate(zip(bboxes, confs))]
            response = LayoutImageResponse(
                image_name=basename(result.path),
                regions=regions.copy(),
            )
            ret.append(response)
        return ret

    async def openseg_infer(self, folder_path: str, language: str) -> list[LayoutImageResponse]:
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

    async def craft_infer(self, folder_path: str) -> list[LayoutImageResponse]:
        return await CRAFTProcessor()(folder_path)

    async def merge_ajoy_openseg(
        self, ajoy_result: list[LayoutImageResponse],
        openseg_result: list[LayoutImageResponse]
    ) -> list[LayoutImageResponse]:
        ajoy_data = [i.model_dump() for i in ajoy_result]
        openseg_data = [i.model_dump() for i in openseg_result]
        merged_data = merge_ajoy_openseg(openseg_data, ajoy_data)
        return [LayoutImageResponse.model_validate(i) for i in merged_data]

    async def merge_ajoy_openseg_craft(
        self, ajoy_result: list[LayoutImageResponse],
        openseg_result: list[LayoutImageResponse],
        craft_result: list[LayoutImageResponse]
    ) -> list[LayoutImageResponse]:
        ajoy_data = [i.model_dump() for i in ajoy_result]
        openseg_data = [i.model_dump() for i in openseg_result]
        craft_data = [i.model_dump() for i in craft_result]
        if self.model == 'V-05.02':
            merged_data = merge_all_regions(openseg_data, ajoy_data, craft_data)
        else:
            merged_data = merge_3_new(openseg_data, ajoy_data, craft_data)
        return [LayoutImageResponse.model_validate(i) for i in merged_data]

    async def __call__(self, folder_path: str, **kwargs: Any):
        language = kwargs.get('language', 'english')
        ajoy_result = await self.ajoy_infer(folder_path)
        openseg_result = await self.openseg_infer(folder_path, language)
        if self.model == 'V-05.01':
            return await self.merge_ajoy_openseg(ajoy_result, openseg_result)
        else:
            craft_result = await self.craft_infer(folder_path)
            return await self.merge_ajoy_openseg_craft(
                ajoy_result, openseg_result, craft_result
            )

register(ModelChoice.v0501)(V05xxProcessor(model='V-05.01'))
register(ModelChoice.v0502)(V05xxProcessor(model='V-05.02'))
register(ModelChoice.v0503)(V05xxProcessor(model='V-05.03'))