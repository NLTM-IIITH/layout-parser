"""
Helper functions for preprocessing

Author: Vignesh E
"""

import os
import shutil
from os.path import join
from typing import List, Tuple
import time

import json

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from fastapi import UploadFile

from ..core.config import IMAGE_FOLDER
from ...models.text_attributes import TextAttributes
from server.modules.main.helper import logtime

from .models import FontAttributeImage, FontAttributesResponse, FontRegion, BoundingBox

def save_uploaded_images(images: List[UploadFile]) -> str:
	print('removing all the previous uploaded files from the image folder')
	# For linux
	# os.system(f'rm -rf {IMAGE_FOLDER}/*')
	os.system(f'del /s /q {IMAGE_FOLDER}\\*.*')
	print(f'Saving {len(images)} to location: {IMAGE_FOLDER}')
	for image in images:
		location = join(IMAGE_FOLDER, f'{image.filename}')
		with open(location, 'wb') as f:
			shutil.copyfileobj(image.file, f)
	return IMAGE_FOLDER

def convert_geometry_to_bbox(
	geometry: Tuple[Tuple[float, float], Tuple[float, float]],
	dim: Tuple[int, int],
	padding: int = 0
) -> BoundingBox:
	"""
	converts the geometry that is fetched from the doctr models
	to the standard bounding box model
	format of the geometry is ((Xmin, Ymin), (Xmax, Ymax))
	format of the dim is (height, width)
	"""
	x1 = int(geometry[0][0] * dim[1])
	y1 = int(geometry[0][1] * dim[0])
	x2 = int(geometry[1][0] * dim[1])
	y2 = int(geometry[1][1] * dim[0])
	return BoundingBox(
		x=x1 - padding,
		y=y1 - padding,
		w=x2-x1 + padding,
		h=y2-y1 + padding,
	)

def process_image_font_attributes_doctr(image_path: str,k_size:int, bold_threshold: float) -> List[FontAttributeImage]:
	"""
	Performs word left font attribute detection using doctr.

	Returns a list of FontAttributeImage
	"""
	t = time.time()
	predictor = ocr_predictor(pretrained=True)
	logtime(t, 'Time taken to load the doctr model')

	files = [join(image_path, i) for i in os.listdir(image_path)]
	doc = DocumentFile.from_images(files)

	t = time.time()
	a = predictor(doc)
	ta = TextAttributes(files,'doctr',thres=bold_threshold,k_size=k_size)
	ta_out = ta.generate(a.export(),'json')
	logtime(t, 'Time taken to perform doctr inference')

	t = time.time()
	result = []

	for p,page in enumerate(ta_out['pages']):
		dim = page['dimensions']
		fontregions = []
		for b,block in enumerate(page['blocks']):
			for l,line in enumerate(block['lines']):
				for w,word in enumerate(line['words']):
					fontregions.append(
						FontRegion(
							bounding_box= convert_geometry_to_bbox(word["geometry"],dim),
							fontColor = word['color'],
							fontDecoration = word['font_weight'],
							fontSize=None
						)
					)
		result.append(
			FontAttributeImage(
				image=os.path.basename(files[p]),
				font_regions=fontregions.copy()
			)
		)
	logtime(t, 'Time taken to generate result')
	return result


def process_image_font_attributes_tesseract(image_path: str):
	"""
	Performs word left font attribute detection using tesseract.

	Returns a list of FontAttributeImage
	"""
	pass
