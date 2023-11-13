import os
import shutil
import time
import uuid
from collections import OrderedDict
from os.path import join
from subprocess import check_output, run
from tempfile import TemporaryDirectory
from typing import List, Tuple

import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import UploadFile

from ..core.config import IMAGE_FOLDER
from .models import *

# TODO: remove this line and try to set the env from the docker-compose file.
os.environ['USE_TORCH'] = '1'

def logtime(t: float, msg:  str) -> None:
	print(f'[{int(time.time() - t)}s]\t {msg}')

t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREDICTOR_V2 = ocr_predictor(pretrained=True)
if os.path.exists('/models/model.pt'):
	state_dict = torch.load('/models/model.pt')
	
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	PREDICTOR_V2.det_predictor.model.load_state_dict(new_state_dict)
logtime(t, 'Time taken to load the doctr model')


def save_uploaded_image(image: UploadFile) -> str:
	"""
	function to save the uploaded image to the disk

	@returns the absolute location of the saved image
	"""
	t = time.time()
	print('removing all the previous uploaded files from the image folder')
	os.system(f'rm -rf {IMAGE_FOLDER}/*')
	location = join(IMAGE_FOLDER, '{}.{}'.format(
		str(uuid.uuid4()),
		image.filename.strip().split('.')[-1]
	))
	with open(location, 'wb+') as f:
		shutil.copyfileobj(image.file, f)
	logtime(t, 'Time took to save one image')
	return location

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
			'-v', f'{IMAGE_FOLDER}:/data',
			'parser:craft',
			'python', 'test.py'
		])

def process_multiple_image_doctr_v2(folder_path: str) -> List[LayoutImageResponse]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""

	files = [join(folder_path, i) for i in os.listdir(folder_path)]
	t = time.time()
	doc = DocumentFile.from_images(files)
	logtime(t, 'Time taken to load all the images')

	t = time.time()
	a = PREDICTOR_V2(doc)
	logtime(t, 'Time taken to perform doctr inference')

	t = time.time()
	ret = []
	for idx in range(len(files)):
		page = a.pages[idx]
		# in the format (height, width)
		dim = page.dimensions
		lines = []
		for i in page.blocks:
			lines += i.lines
		regions = []
		for i, line in enumerate(lines):
			for word in line.words:
				regions.append(
					Region.from_bounding_box(
						convert_geometry_to_bbox(word.geometry, dim, padding=5),
						line=i+1,
					)
				)
		ret.append(
			LayoutImageResponse(
				regions=regions.copy(),
				image_name=os.path.basename(files[idx])
			)
		)
	logtime(t, 'Time taken to process the doctr output')
	return ret