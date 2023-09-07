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
from .readingOrder import *

# TODO: remove this line and try to set the env from the docker-compose file.
os.environ['USE_TORCH'] = '1'

def logtime(t: float, msg:  str) -> None:
	print(f'[{int(time.time() - t)}s]\t {msg}')

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
		# TODO: add the proper error detection if the txt file is not found
		# image_name = os.path.basename(file).strip()[4:].replace('txt', 'jpg')
		img_name = os.path.basename(file).strip()[4:].replace('.txt', '')
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
		# TODO: add the proper error detection if the txt file is not found
		# image_name = os.path.basename(file).strip()[4:].replace('txt', 'jpg')
		img_name = os.path.basename(file).strip()[4:].replace('.txt', '')
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
	t = time.time()
	predictor = ocr_predictor(pretrained=True)
	logtime(t, 'Time taken to load the doctr model')

	files = [join(folder_path, i) for i in os.listdir(folder_path)]
	doc = DocumentFile.from_images(files)

	t = time.time()
	a = predictor(doc)
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
				image_name=os.path.basename(files[idx])
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
				image_name=os.path.basename(files[idx])
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
		predictor = ocr_predictor(pretrained=True)
		a = predictor(doc)
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

####READING ORDER
def doctr_predictions(directory):
#     #Gets the predictions from the model
    
	doc = DocumentFile.from_images(directory)
	result = PREDICTOR_V2(doc)
	dic = result.export()
	
	page_dims = [page['dimensions'] for page in dic['pages']]
	
	regions = []
	abs_coords = []
	
	regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
	abs_coords = [
	[[int(round(word['geometry'][0][0] * dims[1])), 
	int(round(word['geometry'][0][1] * dims[0])), 
	int(round(word['geometry'][1][0] * dims[1])), 
	int(round(word['geometry'][1][1] * dims[0]))] for word in words]
	for words, dims in zip(regions, page_dims)
	]
	
	return abs_coords

def Reading_Order_Generator(image_file):

	pred = doctr_predictions(image_file)
	df = pd.DataFrame(pred)
	df = df.T
	# df = pd.read_json(doctr_file) #commented - to avoid saving doctr output as json
	# df = df.T #commented - to avoid saving doctr output as json
	euclidean = pd.DataFrame()
	kde = pd.DataFrame()
	calculate_center_points(df, kde)
	horizontal_neighbors, vertical_neighbors = find_closest_neighbors(kde)
	x = kde_estimate(horizontal_neighbors)
	y = kde_estimate(vertical_neighbors)
	calculate_center_points(df,euclidean)
	calculate_rightbox(euclidean,x)
	calculate_leftbox(euclidean,x)
	calculate_topbox(euclidean,y)
	calculate_bottombox(euclidean,y)
	
	G = create_graphs(euclidean)
	paragraphs_json = get_paras(G)
	
	component = pd.DataFrame()
	
	# euclidean = pd.read_csv('connections.csv') #commented - to avoid saving euclidean as csv 
	# d1 = pd.read_json('paragraph.json') #commented - to avoid saving get_paras() output as json
	d1 = pd.DataFrame(paragraphs_json)
	target_components = d1.values.tolist()
	# print(target_components)
	image = cv2.imread(image_file)
	component = recognise_paragraphs(image, target_components, euclidean, image_file)
	# print(component)
	min_idx =  minimum_euclidean(component)
	# print(min_idx)
	component = paragraph_order(component)
	# visualise_paragraph_order(image, target_components, euclidean,component)
	new = pd.DataFrame()
	sort_order = component.sort_values('Order').index
	new = component.loc[sort_order]
	new = new.reset_index(drop=True)
	new_euclidean = pd.DataFrame()
	euclidean = word_order(new, euclidean)
	sort_order = euclidean.sort_values('Order').index
	new_euclidean = euclidean.loc[sort_order]
	# print(new_euclidean)
	new_euclidean = new_euclidean.reset_index(drop=True)
	# print(new_euclidean)
	image = cv2.imread(image_file)
	image_with_boxes, reading_order_json = reading_order(image,new_euclidean, image_file)
	# output_path = 'Output.png'
	# cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
	# euclidean.to_csv('Euclidean.csv')
	# print(sort_order)
	# print(reading_order_json)
	return image_with_boxes, reading_order_json
	# print(new_euclidean)

def process_multiple_pages_ReadingOrderGenerator(folder_path: str) -> List[LayoutImageResponse]:
	files = [join(folder_path, i) for i in os.listdir(folder_path)]
	ret = []
	for idx in range(len(files)):
		reading_order_image, reading_order = Reading_Order_Generator(files[idx])
		ret.append(LayoutImageResponse(regions=reading_order.copy(),image_name=os.path.basename(files[idx])))

	logtime(t, 'Time taken to generate Reading Order')
	return ret		
		
	
