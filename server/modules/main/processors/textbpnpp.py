import os
import time
from os.path import basename, join
from subprocess import run
from typing import Any

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

class TextBPNPPProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        """
        Given a path to the folder if images, this function returns a list
        of word level bounding boxes of all the images
        """
        t = time.time()
        run((
            'docker run --rm --net host '
            f'--gpus all -v {folder_path}:/data '
            'layout:textbpnpp /app/.venv/bin/python infer.py'
        ).strip().split(' '))
        logtime(t, 'Time took to run the textbpnpp docker container')
        img_files = [i for i in os.listdir(folder_path) if not i.endswith('txt')]
        files = [join(folder_path, i) for i in os.listdir(folder_path) if i.endswith('txt')]
        print('files = ', files)
        ret = []
        t = time.time()
        for file in files:
            img_name = basename(file).strip().replace('.txt', '')
            image_name = [i for i in img_files if os.path.splitext(i)[0] == img_name][0]
            print(file, img_name, image_name)
            a = open(file, 'r').read().strip().split('\n')
            a = [i.strip() for i in a if i]
            print('a = ', a)
            regions = []
            for i in a:
                word = i.strip().split(',')
                word = list(map(int, word))
                regions.append(
                    Region.from_bounding_box(
                        BoundingBox(
                            x=word[0],
                            y=word[1],
                            w=word[2],
                            h=word[3],
                        ),
                        line=1
                    )
                )
            ret.append(
                LayoutImageResponse(
                    image_name=image_name,
                    regions=regions.copy()
                )
            )
        logtime(t, 'Time took to process the output of the textbpnpp docker')
        return ret

register(ModelChoice.textbpnpp)(TextBPNPPProcessor())