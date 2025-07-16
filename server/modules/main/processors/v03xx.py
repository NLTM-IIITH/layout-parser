import time
from os.path import basename
from typing import Any

from ultralytics import YOLO

from server.config import settings

from ..factory import register
from ..models import LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

class V03xxProcessor:

    def __init__(self, model: str = 'V-03.01') -> None:
        self.model = model
        return super().__init__()

    async def __call__(self, folder_path: str, **kwargs: Any):
        model_iden = int(self.model.split('.')[-1])
        model_path = settings.v03xx_model_path/ f'{model_iden}.pt'
        model = YOLO(model_path)
        results = model.predict(source=folder_path, max_det=300)
        ret = []
        for result in results:
            bboxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist() # type: ignore
            bboxes = [tuple(i) for i in bboxes]
            confs = (result.boxes.conf.cpu().numpy().round(3)) # type: ignore
            regions = [Region.from_xyxy(v, line=i+1, conf=c) for i, (v, c) in enumerate(zip(bboxes, confs))]
            ret.append(
                LayoutImageResponse(
                    image_name=basename(result.path),
                    regions=regions.copy(),
                )
            )
        return ret

register(ModelChoice.v0301)(V03xxProcessor(model='V-03.01'))
register(ModelChoice.v0302)(V03xxProcessor(model='V-03.02'))
register(ModelChoice.v0303)(V03xxProcessor(model='V-03.03'))
register(ModelChoice.v0304)(V03xxProcessor(model='V-03.04'))