import os
import time
from os.path import basename, join
from typing import Any

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

class DocTRProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        """
        given the path of the image, this function returns a list
        of bounding boxes of all the word detected regions.

        @returns list of BoundingBox class
        """
        t = time.time()
        predictor = ocr_predictor(pretrained=True)
        logtime(t, 'Time taken to load the doctr model')

        files = [join(folder_path, i) for i in os.listdir(folder_path)]
        doc = DocumentFile.from_images(files)

        t = time.time()
        a = predictor(doc)
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
            order = 1
            for i, line in enumerate(lines):
                for word in line.words:
                    regions.append(
                        Region.from_bounding_box(
                            convert_geometry_to_bbox(word.geometry, dim),
                            line=i+1,
                            order=order
                        )
                    )
                    order += 1
            ret.append(
                LayoutImageResponse(
                    regions=regions.copy(),
                    image_name=basename(files[idx])
                )
            )
        logtime(t, 'Time taken to process the doctr output')
        return ret

register(ModelChoice.doctr)(DocTRProcessor())