import json
import time
from os.path import join
from subprocess import check_output
from typing import Any

from fastapi import HTTPException

from server.config import settings

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

class IndicDLPProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        try:
            check_output([
                'docker',
                'run',
                '--rm',
                '--gpus', 'all',
                '-v', f'{folder_path}:/data',
                '-v', f'{settings.indicdlp_model_path}:/model:ro',
                'layout:indicdlp'
            ])
            with open(join(folder_path, 'out.json'), 'r') as f:
                a = json.loads(f.read().strip())
            ret = []
            for page in a.keys():
                regions = []
                for bbox in a[page]:
                    regions.append(
                        Region.from_bounding_box(
                            BoundingBox(
                                x=int(bbox['x']),
                                y=int(bbox['y']),
                                w=int(bbox['w']),
                                h=int(bbox['h']),
                            ),
                            label=bbox['label'],
                            confidence=bbox.get('confidence', 0.0)
                        )
                    )
                ret.append(
                    LayoutImageResponse(
                      image_name=page,
                      regions=regions.copy()
                    )
                )
            return ret     
        except Exception as e:
            print(e)
            raise HTTPException(500, 'Error while calling IndicDLP model')

register(ModelChoice.indicdlp)(IndicDLPProcessor())