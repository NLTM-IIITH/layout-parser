import os
import shutil
import uuid
from os.path import join
from typing import List, Tuple

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import UploadFile

from .config import IMAGE_FOLDER
from .models import *

# TODO: remove this line and try to set the env from the docker-compose file.
os.environ['USE_TORCH'] = '1'


def save_uploaded_image(image: UploadFile) -> str:
	"""
	function to save the uploaded image to the disk

	@returns the absolute location of the saved image
	"""
	location = join(IMAGE_FOLDER, '{}.{}'.format(
		str(uuid.uuid4()),
		image.filename.strip().split('.')[-1]
	))
	with open(location, 'wb+') as f:
		shutil.copyfileobj(image.file, f)
	return location

def convert_geometry_to_bbox(
	geometry: Tuple[Tuple[float, float], Tuple[float, float]],
	dim: Tuple[int, int]
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
		x=x1,
		y=y1,
		w=x2-x1,
		h=y2-y1,
	)

def process_image(image_path: str) -> List[Region]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	print('running the doctr model for image...', end='')
	predictor = ocr_predictor(pretrained=True)
	doc = DocumentFile.from_images(image_path)
	a = predictor(doc)
	print('done')
	a = a.pages[0]
	# in the format (height, width)
	dim = a.dimensions
	lines = []
	for i in a.blocks:
		lines += i.lines
	ret = []
	for i, line in enumerate(lines):
		for word in line.words:
			ret.append(
				Region.from_bounding_box(
					convert_geometry_to_bbox(word.geometry, dim),
					line=i+1,
				)
			)
	return ret
