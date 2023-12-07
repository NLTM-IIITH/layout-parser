import os
import shutil
import time
import uuid
from collections import OrderedDict
from os.path import basename, join
from subprocess import check_output, run
from tempfile import TemporaryDirectory
from typing import List, Tuple

import pytesseract
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import UploadFile
from skimage.filters import threshold_otsu, threshold_sauvola

from ..core.config import IMAGE_FOLDER, TESS_LANG
from .models import *
from .readingOrder import *
from .croppadfix import *

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

def binarize_image_otsu(image):
	threshold = threshold_otsu(image)
	image1 = image < threshold
	return image1

def binarize_image_sauvola(image):
	threshold = threshold_sauvola(image)
	image1 = image < threshold
	return image1 

def horizontal_projections(image):
	return np.sum(image, axis=1)

def find_peaks_valley(hpp):
	line_index = []
	i = 0
	prev_i = -1
	while(i<len(hpp)-1):
		print("i==>",i)
		index1 = i
		flag1 = 0
		flag2 = 0
		for j in range (i, len(hpp)-1, 1):
			if (hpp[j] != 0):
				index1 = j-1
				line_index.append(index1)
				flag1 = 1
				break
		for j in range (index1+1, len(hpp)-1, 1):
			if (hpp[j] == 0 and flag1 ==1):
				index2 = j
				line_index.append(index2)
				flag2 = 1
				break
		if (flag1 ==1 and flag2==1):
			i = index2	
		if (flag1 == 0 and flag2 ==0):
			break
		if i == prev_i:
			break
		prev_i = i
	return line_index		

def process_multiple_tesseract(folder_path: str, language: str) -> List[LayoutImageResponse]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	ret = []
	for filename in os.listdir(folder_path):
		image_path = join(folder_path, filename)
		image = cv2.imread(image_path)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		lang = TESS_LANG.get(language, 'eng')
		print(lang)
		results = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, lang=lang)
		print(results)

		regions = []
		line = 1
		for i in range(0, len(results['text'])):
			if int(results['conf'][i]) <= 0:
				# Skipping the region as confidence is too low
				continue
			x = results['left'][i]
			y = results['top'][i]
			w = results['width'][i]
			h = results['height'][i]
			regions.append(Region(
				bounding_box=BoundingBox(
					x=x,
					y=y,
					w=w,
					h=h
				),
				line=results['line_num'][i] + 1,
			))
			line += 1
		ret.append(LayoutImageResponse(
			image_name=basename(image_path),
			regions=regions.copy()
		))
	return ret

def process_multiple_urdu_v1(folder_path: str) -> List[LayoutImageResponse]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	ret = []
	for filename in os.listdir(folder_path):
		# image_path = folder_path + filename
		image_path = join(folder_path, filename)
		image = cv2.imread(image_path)
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, image_w = image_gray.shape
		image_thres = binarize_image_sauvola(image_gray)
		print('binarize image completed')
		hpp = horizontal_projections(image_thres)
		print('horizontal projections completed')

		#finding vally and peak of histigram
		line_index = find_peaks_valley(hpp)
		print('found the peaks and valleys')

		regions = []
		line = 1
		for i in range(0,len(line_index)-1,2):
			y1 = int(line_index[i])
			y2 = int(line_index[i+1])
			x1 = 0
			x2 = image_w
			# color = (0,0,255)
			# thickness = 2 
			if (y2-y1>10):
				regions.append(Region(
					bounding_box=BoundingBox(
						x=x1,
						y=y1,
						w=x2-x1,
						h=y2-y1
					),
					line=line,
				))
				line += 1
		ret.append(LayoutImageResponse(
			image_name=basename(image_path),
			regions=regions.copy()
		))
	return ret

def process_image_urdu_v1(image_path: str) -> List[Region]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	folder_path = os.path.dirname(image_path)
	ret = []
	for filename in os.listdir(folder_path):
		image_path = join(folder_path, filename)
		print(image_path)
		image = cv2.imread(image_path)
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, image_w = image_gray.shape
		image_thres = binarize_image_sauvola(image_gray)
		hpp = horizontal_projections(image_thres)

		#finding vally and peak of histigram
		line_index = find_peaks_valley(hpp)

		regions = []
		line = 1
		for i in range(0,len(line_index)-1,2):
			y1 = int(line_index[i])
			y2 = int(line_index[i+1])
			x1 = 0
			x2 = image_w
			# color = (0,0,255)
			# thickness = 2 
			if (y2-y1>10):
				regions.append(Region(
					bounding_box=BoundingBox(
						x=x1,
						y=y1,
						w=x2-x1,
						h=y2-y1
					),
					line=line,
				))
				line += 1
		ret.append(LayoutImageResponse(
			image_name=basename(image_path),
			regions=regions.copy()
		))
	return ret[0].regions


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
	try:
		a = a[0]
		a = open(a, 'r').read().strip()
		a = a.split('\n\n')
		a = [i.strip().split('\n') for i in a]
	except:
		a = []
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

def Reading_Order_Generator(image_file, width_p, header_p, footer_p, para_only,col_only):

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
	component = recognise_paragraphs(image, target_components, euclidean, image_file, width_p, header_p, footer_p)
	# print(component)
	min_idx =  minimum_euclidean(component)
	# print(min_idx)
	component = paragraph_order(component)

	if para_only is True and col_only is False:
		img = visualise_paragraph_order(image, target_components, euclidean,component)
		return img
	elif para_only is False and col_only is True:
		img = get_col(image,component)
		return img
	elif para_only is True and col_only is True:
		pass
	elif para_only is False and col_only is False:	
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
		image_with_boxes, reading_order_json = reading_order(image,new_euclidean, image_file, header_p, footer_p)
		# output_path = 'Output.png'
		# cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
		# euclidean.to_csv('Euclidean.csv')
		# print(sort_order)
		# print(reading_order_json)
		return image_with_boxes, reading_order_json
		# print(new_euclidean)

def process_multiple_pages_ReadingOrderGenerator(folder_path: str, left_right_percentages: int, header_percentage: int, footer_percentage: int) -> List[LayoutImageResponse]:
	files = [join(folder_path, i) for i in os.listdir(folder_path)]
	ret = []
	para_only = False #para_only = True when visualizing para boxes, here we get only the reading_order json, so para_only not required
	col_only = False
	for idx in range(len(files)):
		reading_order_image, reading_order = Reading_Order_Generator(files[idx], left_right_percentages, header_percentage, footer_percentage, para_only,col_only)
		ret.append(LayoutImageResponse(regions=reading_order.copy(),image_name=basename(files[idx])))

	logtime(t, 'Time taken to generate Reading Order')
	return ret		

#croppadfix
def cropPadFix(image_file):
	# visualize_preds_dir(args.ImageFile)
	cropped_TL, img_cropped = save_cropped(image_file)
	# visualize_preds_dir('/home2/sreevatsa/cropped.png')
	img = visualized_rescaled_bboxes_from_cropped(img_cropped,image_file,cropped_TL)
	return img
		
