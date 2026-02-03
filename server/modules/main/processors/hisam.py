import asyncio
import json
import os
import time
from os.path import join
from subprocess import check_output
from typing import Any

import aiohttp
from fastapi import HTTPException

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')


async def submit_jobs(folder_path: str) -> str:
    url = "http://10.4.16.102:5016/api/v1/detections/process/"
    headers = {
        "accept": "application/json",
        "X-API-Token": "usk_q030qJvLnnn8ZQa8vJjDKwijeF_H3klYNzSN_mJjo6X8-2a_TlAR5w",
    }
    files = [("files", open(join(folder_path, i), "rb")) for i in os.listdir(folder_path) if i.endswith(('.png', '.jpg', '.jpeg'))]
    ret = None

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        for name, file_obj in files:
            form.add_field(
                name,
                file_obj,
                filename=file_obj.name,
                content_type="image/jpeg"
            )

        async with session.post(url, headers=headers, data=form) as response:
            ret = await response.json()

    # Close files manually
    for _, f in files:
        f.close()

    return ret.get('job_id', '')

async def check_job_status(job_id: str):
    url = f'http://10.4.16.102:5016/api/v1/detections/status/{job_id}'
    headers = {
        "X-API-Token": "usk_q030qJvLnnn8ZQa8vJjDKwijeF_H3klYNzSN_mJjo6X8-2a_TlAR5w",
    }
    status = 'pending'
    count = 1
    async with aiohttp.ClientSession() as session:
        while status != 'COMPLETED' and count <= 60:
            async with session.get(url, headers=headers) as response:
                ret = await response.json()
                status = ret.get('status', 'pending')
                print(f"Hi-Sam: Job {job_id} status: {status}")
            if status != 'COMPLETED':
                await asyncio.sleep(3)  # wait 5 seconds before checking again
                count += 1
    return status

async def get_job_result(job_id: str):
    url = f'http://10.4.16.102:5016/api/v1/detections/status/hisam_result/{job_id}'
    headers = {
        "X-API-Token": "usk_q030qJvLnnn8ZQa8vJjDKwijeF_H3klYNzSN_mJjo6X8-2a_TlAR5w",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response = await response.json()
    return response


class HisamProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        try:
            job = await submit_jobs(folder_path)
            await check_job_status(job)
            ret = await get_job_result(job)
            rescale_factor = ret['results'][0]['rescale_factor']
            words = ret['results'][0]['output']['paragraphs']
            regions = []
            for i in words:
                regions.append(Region.from_xyxy(tuple(map(lambda x: int(x / rescale_factor), i))))
            return [LayoutImageResponse(image_name='hisam_output', regions=regions)]
            with open('/home/layout/hisam.json', 'w', encoding='utf-8') as f:
                json.dump(ret, f, indent=4)
            return ret     
        except Exception as e:
            print(e)
            raise e
            raise HTTPException(500, 'Error while calling Textron model')

register(ModelChoice.hisam)(HisamProcessor())