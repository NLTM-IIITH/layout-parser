import json
from os.path import join
from subprocess import check_output
from typing import Any

from ..factory import register
from ..models import LayoutImageResponse, ModelChoice, Region


class V06xxProcessor:

    async def __call__(self, folder_path: str, **kwargs: Any):
        check_output([
            'docker',
            'run',
            '--rm',
            '--gpus', 'all',
            '-v', f'{folder_path}:/data',
            'layout:v0601'
        ])
        with open(join(folder_path, 'out.json'), 'r') as f:
            a = json.loads(f.read().strip())
        ret = []
        for each in a:
            ret.append(LayoutImageResponse(
                image_name=each['image_name'],
                regions = [Region.from_xyxy(**i) for i in each['regions']]
            ))
        return ret

register(ModelChoice.v0601)(V06xxProcessor())