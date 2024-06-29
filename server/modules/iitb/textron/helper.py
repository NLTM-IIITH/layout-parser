import json
import uuid
from os.path import dirname, join
from subprocess import check_output

import cv2

from .models import BoundingBox, LayoutImageResponse, Region


def run_docker(folder):
    check_output([
        'docker',
        'run',
        '--rm',
        '--net',
        'host',
        '-v', f'{folder}:/data',
        'layout:iitb-textron'
    ])

def process_textron_output(folder_path: str) -> list[LayoutImageResponse]:
    try:
        run_docker(folder_path)
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
                            label=bbox['label'],
                        )
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

def textron_visualize(image_path: str) -> str:
    run_docker(dirname(image_path))
    with open(join(dirname(image_path), 'out.json'), 'r') as f:
        a = json.loads(f.read().strip())
    for page in a.keys():
        regions=[]
        for bbox in a[page]:
            regions.append(
                Region.from_bounding_box(
                    BoundingBox(
                        x=int(bbox['x']),
                        y=int(bbox['y']),
                        w=int(bbox['w']),
                        h=int(bbox['h']),
                        label=bbox['label'],
                    )
                )
            )
        break
    save_location = join(dirname(image_path), f'{str(uuid.uuid4())}.jpg')

    bboxes = [i.bounding_box for i in regions]
    bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
    img = cv2.imread(image_path)
    count = 1
    for i in bboxes:
        img = cv2.rectangle(img, i[0], i[1], (0,0,255), 3)
        img = cv2.putText(
            img,
            str(count),
            (i[0][0]-5, i[0][1]-5),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,0,255),
            1,
            cv2.LINE_AA
        )
        count += 1
    cv2.imwrite(save_location, img)
    return save_location
