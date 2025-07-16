import json
import time
from os.path import join
from subprocess import check_output
from typing import Any

from fastapi import HTTPException

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

class TextronProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        try:
            check_output([
                'docker',
                'run',
                '--rm',
                '--net',
                'host',
                '-v', f'{folder_path}:/data',
                'layout:iitb-textron'
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
            raise HTTPException(500, 'Error while calling Textron model')

register(ModelChoice.textron)(TextronProcessor())