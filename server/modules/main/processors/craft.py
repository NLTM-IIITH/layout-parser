import os
import time
from os.path import basename, join
from subprocess import check_output, run
from typing import Any

from server.config import settings

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

def load_craft_container():
    command = 'docker container ls --format "{{.Names}}"'
    a = check_output(command, shell=True).decode('utf-8').strip().split('\n')
    if 'parser-craft' not in a:
        print('CRAFT docker container not found! Starting new container')
        run([
            'docker',
            'run', '-d',
            '--name=parser-craft',
            '--gpus', 'all',
            '-v', f'{settings.image_folder}:/data',
            'parser:craft',
            'python', 'test.py'
        ])

class CRAFTProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        """
        Given a path to the folder if images, this function returns a list
        of word level bounding boxes of all the images
        """
        t = time.time()
        load_craft_container()
        run([
            'docker',
            'exec', 'parser-craft',
            'bash', 'infer.sh'
        ])
        logtime(t, 'Time took to run the craft docker container')
        img_files = [i for i in os.listdir(folder_path) if not i.endswith('txt')]
        files = [join(folder_path, i) for i in os.listdir(folder_path) if i.endswith('txt')]
        ret = []
        t = time.time()
        for file in files:
            # TODO: add the proper error detection if the txt file is not found
            img_name = basename(file).strip()[4:].replace('.txt', '')
            image_name = [i for i in img_files if os.path.splitext(i)[0] == img_name][0]
            a = open(file, 'r').read().strip()
            a = a.split('\n\n')
            a = [i.strip().split('\n') for i in a]
            regions = []
            order = 1
            for i, line in enumerate(a):
                for j in line:
                    word = j.strip().split(',')
                    word = list(map(int, word))
                    regions.append(
                        Region.from_bounding_box(
                            BoundingBox(
                                x=word[0],
                                y=word[1],
                                w=word[2],
                                h=word[3],
                            ),
                            line=i+1,
                            order=order,
                        )
                    )
                    order += 1
            ret.append(
                LayoutImageResponse(
                    image_name=image_name,
                    regions=regions.copy()
                )
            )
        logtime(t, 'Time took to process the output of the craft docker')
        return ret

register(ModelChoice.craft)(CRAFTProcessor())