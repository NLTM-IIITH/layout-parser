import os
import shutil
import time
import uuid
from collections import OrderedDict
from os.path import basename, join
from subprocess import check_output, run
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
PREDICTOR = ocr_predictor(pretrained=True)
logtime(t, 'Time taken to load the doctr model')

t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREDICTOR_V2 = ocr_predictor(pretrained=True).to(device)
if os.path.exists('/home/layout/models/v2_doctr/model.pt'):
	state_dict = torch.load('/home/layout/models/v2_doctr/model.pt')
	
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

def process_multiple_image_craft(folder_path: str) -> List[LayoutImageResponse]:
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
		# [4:] is added because craft prefix the text filenames with res_
		img_name = basename(file).strip()[4:].replace('.txt', '')
		image_name = [i for i in img_files if os.path.splitext(i)[0] == img_name][0]
		print(image_name)
		a = open(file, 'r').read().strip()
		a = a.split('\n\n')
		a = [i.strip().split('\n') for i in a]
		regions = []
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
						line=i+1
					)
				)
		ret.append(
			LayoutImageResponse(
				image_name=image_name,
				regions=regions.copy()
			)
		)
	logtime(t, 'Time took to process the output of the craft docker')
	return ret

def run_textpms_container():
	run([
		'docker',
		'run', '-it', '--rm',
		'--name=parser-textpms',
		'--gpus', 'all',
		'-v', f'{IMAGE_FOLDER}:/data',
		'-v', f'/home/layout/models/TextPMs:/model:ro',
		'parser:textpms',
		'python', 'infer.py'
	])


def process_multiple_image_textpms(folder_path: str) -> List[LayoutImageResponse]:
	"""
	Given a path to the folder if images, this function returns a list
	of word level bounding boxes of all the images
	"""
	t = time.time()
	run_textpms_container()
	logtime(t, 'Time took to run the textpms docker container')
	img_files = [i for i in os.listdir(folder_path) if not i.endswith('txt')]
	files = [join(folder_path, i) for i in os.listdir(folder_path) if i.endswith('txt')]
	ret = []
	t = time.time()
	for file in files:
		img_name = basename(file).strip().replace('.txt', '')
		image_name = [i for i in img_files if os.path.splitext(i)[0] == img_name][0]
		print(image_name)
		a = open(file, 'r').read().strip().split('\n')
		a = [i.strip() for i in a]
		regions = []
		for i in a:
			# i is in the format x,y;x,y;x,y...
			regions.append(
				PolygonRegion.from_points(
					points=[tuple(map(int, j.split(','))) for j in i.split(';')]
				)
			)
		ret.append(
			LayoutImageResponse(
				image_name=image_name,
				regions=regions.copy()
			)
		)
	logtime(t, 'Time took to process the output of the craft docker')
	return ret


def process_multiple_image_worddetector(folder_path: str) -> List[LayoutImageResponse]:
	"""
	Given a path to the folder if images, this function returns a list
	of word level bounding boxes of all the images
	"""
	t = time.time()
	command = f'docker run -it --rm --gpus all -v {folder_path}:/data parser:worddetector python infer.py'
	check_output(command, shell=True)
	logtime(t, 'Time took to run the worddetector docker container')
	img_files = [i for i in os.listdir(folder_path) if not i.endswith('txt')]
	files = [join(folder_path, i) for i in os.listdir(folder_path) if i.endswith('txt')]
	ret = []
	t = time.time()
	for file in files:
		img_name = basename(file).strip()[4:].replace('.txt', '')
		image_name = [i for i in img_files if os.path.splitext(i)[0] == img_name][0]
		print(image_name)
		a = open(file, 'r').read().strip()
		a = a.split('\n\n')
		a = [i.strip().split('\n') for i in a]
		regions = []
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
						line=i+1
					)
				)
		ret.append(
			LayoutImageResponse(
				image_name=image_name,
				regions=regions.copy()
			)
		)
	logtime(t, 'Time took to process the output of the worddetector docker')
	return ret


def process_multiple_image_doctr(folder_path: str) -> List[LayoutImageResponse]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	files = [join(folder_path, i) for i in os.listdir(folder_path)]
	doc = DocumentFile.from_images(files)

	t = time.time()
	a = PREDICTOR(doc)
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
						convert_geometry_to_bbox(word.geometry, dim),
						line=i+1,
					)
				)
		ret.append(
			LayoutImageResponse(
				regions=regions.copy(),
				image_name=basename(files[idx])
			)
		)
	logtime(t, 'Time taken to process the doctr output')
	return ret


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
				image_name=basename(files[idx])
			)
		)
	logtime(t, 'Time taken to process the doctr output')
	return ret


def process_image(image_path: str, model: str='doctr') -> List[Region]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	t = time.time()
	doc = DocumentFile.from_images(image_path)
	logtime(t, 'Time taken to load the image')
	t = time.time()
	if model == 'doctr':
		print('performing doctr')
		a = PREDICTOR(doc)
	else:
		print('performing v2_doctr')
		a = PREDICTOR_V2(doc)
	logtime(t, 'Time taken to perform doctr inference')
	t = time.time()
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
					convert_geometry_to_bbox(word.geometry, dim, padding=5),
					line=i+1,
				)
			)
	logtime(t, 'Time taken to process the doctr output')
	return ret


def process_image_craft(image_path: str) -> List[Region]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns: list of BoundingBox class
	"""
	print('running craft model for image...', end='')
	load_craft_container()
	run([
		'docker',
		'exec', 'parser-craft',
		'bash', 'infer.sh'
	])
	a = [join(IMAGE_FOLDER, i) for i in os.listdir(IMAGE_FOLDER) if i.endswith('txt')]
	# TODO: add the proper error detection if the txt file is not found
	a = a[0]
	a = open(a, 'r').read().strip()
	a = a.split('\n\n')
	a = [i.strip().split('\n') for i in a]
	ret = []
	for i, line in enumerate(a):
		for j in line:
			word = j.strip().split(',')
			word = list(map(int, word))
			ret.append(
				Region.from_bounding_box(
					BoundingBox(
						x=word[0],
						y=word[1],
						w=word[2],
						h=word[3],
					),
					line=i+1
				)
			)
	return ret


def process_image_worddetector(image_path: str) -> List[Region]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns: list of BoundingBox class
	"""
	print('running craft model for image...', end='')
	command = f'docker run -it --rm --gpus all -v {IMAGE_FOLDER}:/data parser:worddetector python infer.py'
	check_output(command, shell=True)
	a = [join(IMAGE_FOLDER, i) for i in os.listdir(IMAGE_FOLDER) if i.endswith('txt')]
	# TODO: add the proper error detection if the txt file is not found
	a = a[0]
	a = open(a, 'r').read().strip()
	a = a.split('\n\n')
	a = [i.strip().split('\n') for i in a]
	ret = []
	for i, line in enumerate(a):
		for j in line:
			word = j.strip().split(',')
			word = list(map(int, word))
			ret.append(
				Region.from_bounding_box(
					BoundingBox(
						x=word[0],
						y=word[1],
						w=word[2],
						h=word[3],
					),
					line=i+1
				)
			)
	return ret
