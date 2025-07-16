import os
import time
from collections import OrderedDict
from os.path import basename, join
from typing import Any

import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

def convert_geometry_to_bbox(
    geometry: tuple[tuple[float, float], tuple[float, float]],
    dim: tuple[int, int],
    padding: int = 0
) -> BoundingBox:
    """
    converts the geometry that is fetched from the doctr models
    to the standard bounding box model
    format of the geometry is ((Xmin, Ymin), (Xmax, Ymax))
    format of the dim is (height, width)
    """
    x1 = int(geometry[0][0] * dim[1])
    y1 = int(geometry[0][1] * dim[0])
    x2 = int(geometry[1][0] * dim[1])
    y2 = int(geometry[1][1] * dim[0])
    return BoundingBox(
        x=x1 - padding,
        y=y1 - padding,
        w=x2-x1 + padding,
        h=y2-y1 + padding,
    )

class V2DocTRProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        """
        given the path of the image, this function returns a list
        of bounding boxes of all the word detected regions.

        @returns list of BoundingBox class
        """

        files = [join(folder_path, i) for i in os.listdir(folder_path)]
        t = time.time()
        doc = DocumentFile.from_images(files)
        logtime(t, 'Time taken to load all the images')


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        PREDICTOR_V2 = ocr_predictor(pretrained=True).to(device)
        if os.path.exists('/home/layout/models/v2_doctr/model.pt'):
            state_dict = torch.load('/home/layout/models/v2_doctr/model.pt')

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            PREDICTOR_V2.det_predictor.model.load_state_dict(new_state_dict)

        t = time.time()
        a = PREDICTOR_V2(doc)
        logtime(t, 'Time taken to perform doctr inference')

        t = time.time()
        ret = []
        for idx in range(len(files)):
            page = a.pages[idx]
            # in the format (height, width)
            dim = page.dimensions
            lines = []
            for i in page.blocks:
                lines += i.lines
            regions = []
            for i, line in enumerate(lines):
                for word in line.words:
                    regions.append(
                        Region.from_bounding_box(
                            convert_geometry_to_bbox(word.geometry, dim, padding=5),
                            line=i+1,
                        )
                    )
            ret.append(
                LayoutImageResponse(
                    regions=regions.copy(),
                    image_name=basename(files[idx])
                )
            )
        logtime(t, 'Time taken to process the doctr output')
        del PREDICTOR_V2
        return ret

register(ModelChoice.v2_doctr)(V2DocTRProcessor())