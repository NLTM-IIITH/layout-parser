import base64
import json
import os
from os.path import join
from typing import List

from fastapi import HTTPException

from .models import SIResponse


def process_image_content(image_content: str, savename: str) -> None:
	"""
	input the base64 encoded image and saves the image inside the folder.
	savename is the name of the image to be saved as
	"""
	savefolder = '/home/layout/layout-parser/images'

	assert isinstance(image_content, str)
	with open(join(savefolder, savename), 'wb') as f:
		f.write(base64.b64decode(image_content))


def process_images(images: List[str]):
	"""
	processes all the images in the given list.
	it saves all the images in the /home/ocr/website/images folder and
	returns this absolute path.
	"""
	print('deleting all the previous data from the images folder')
	os.system('rm -rf /home/layout/layout-parser/images/*')
	for idx, image in enumerate(images):
		if image is not None:
			try:
				process_image_content(image, '{}.jpg'.format(idx))
			except:
				raise HTTPException(
					status_code=400,
					detail=f'Error while decodeing and saving the image #{idx}',
				)
		else:
			raise HTTPException(
				status_code=400,
				detail=f'image #{idx} doesnt contain either imageContent or imageUri',
			)


def process_ocr_output() -> List[SIResponse]:
	"""
	process the ./images/out.json file and returns the ocr response.
	"""
	try:
		ret = []
		a = open('/home/layout/layout-parser/images/out.json', 'r').read().strip()
		a = json.loads(a)
		a = list(a.items())
		a = sorted(a, key=lambda x:int(x[0].split('.')[0]))
		return [SIResponse(text=i[1]) for i in a]
	except Exception as e:
		print(e)
		raise HTTPException(
			status_code=500,
			detail='Error while parsing the ocr output'
		)

