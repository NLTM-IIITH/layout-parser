import base64
import json
import os
from os.path import join

from fastapi import HTTPException

from .models import ClassifyResponse


def process_images(images: list[str], path: str='/home/layout/layout-parser/images'):
	"""
	processes all the images in the given list.
	it saves all the images in the /home/ocr/website/images folder and
	returns this absolute path.
	"""
	print('deleting all the previous data from the images folder')
	os.system(f'rm -rf {path}/*')
	for idx, image in enumerate(images):
		if image is not None:
			try:
				# saving the base64 image as JPEG
				assert isinstance(image, str)
				with open(join(path, f'{idx}.jpg'), 'wb') as f:
					f.write(base64.b64decode(image))
			except:
				raise HTTPException(
					status_code=400,
					detail=f'Error while decoding and saving the image #{idx}',
				)
		else:
			raise HTTPException(
				status_code=400,
				detail=f'image #{idx} doesnt contain either imageContent or imageUri',
			)


def process_layout_output(
	path: str = '/home/layout/layout-parser/images'
) -> list[ClassifyResponse]:
	"""
	process the ./images/out.json file and returns the ocr response.
	"""
	try:
		a = open(join(path, 'out.json'), 'r').read().strip()
		a = json.loads(a)
		a = list(a.items())
		a = sorted(a, key=lambda x:int(x[0].split('.')[0]))
		return [ClassifyResponse(text=i[1]) for i in a]
	except Exception as e:
		print(e)
		raise HTTPException(
			status_code=500,
			detail='Error while parsing the ocr output'
		)

