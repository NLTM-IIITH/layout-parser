from os.path import basename
from typing import Any

from ultralytics import YOLO

from server.config import settings

from ..factory import register
from ..models import LayoutImageResponse, ModelChoice, Region


class V04xxProcessor:

    def __init__(self, model: str = 'V-04.01') -> None:
        self.model = model
        return super().__init__()

    async def __call__(self, folder_path: str, **kwargs: Any):
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