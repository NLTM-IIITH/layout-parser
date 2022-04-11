import os
import shutil
import uuid
from os.path import join
from typing import List

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

def process_image(image_path: str) -> List[BoundingBox]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	print('loading the doctr ocr predictor to the memory...', end='')
	predictor = ocr_predictor(pretrained=True)
	print('done')
	ret = []
	doc = DocumentFile.from_images(image_path)
	result = predictor(doc)
	dic = result.export()
	page_words = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
	page_dims = [page['dimensions'] for page in dic['pages']]
	words_abs_coords = [
		[[int(round(word['geometry'][0][0] * dims[1])), int(round(word['geometry'][0][1] * dims[0])), int(round(word['geometry'][1][0] * dims[1])), int(round(word['geometry'][1][1] * dims[0]))] for word in words]
		for words, dims in zip(page_words, page_dims)
	]
	for w in words_abs_coords[0]:
		ret.append(
			BoundingBox(
				x=w[0],
				y=w[1],
				w=w[2]-w[0],
				h=w[3]-w[1],
			)
		)
	return ret

def sort_words(boxes):
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
    
    # boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []
    for box in boxes:
        if box[1] > current_line + mean_height:
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]            
            continue
        tmp_line.append(box)
    lines.append(tmp_line)
        
    for line in lines:
        line.sort(key=lambda box: box[0])
        
    return lines


def sort_regions(regions: List[Region]) -> List[Region]:
	"""
	this function takes the list of Region(s) of a given page and
	sorts then according to the left to right and top to bottom fashion.
	It also determines and assigns a line number to each of the region.

	@returns list of proper sorted Region(s)
	"""
	a = [i.bounding_box for i in regions]
	a = [(i.x,i.y,i.x+i.w,i.y+i.h) for i in a]
	a = sort_words(a)
	print(a)
	region_list = []
	for i, v in enumerate(a):
		for t in sorted(v, key=lambda q:q[0]):
			region_list.append(
				Region(
					bounding_box=BoundingBox(
						x=t[0],
						y=t[1],
						w=t[2]-t[0],
						h=t[3]-t[1],
					),
					# TODO: change the logic such that label is inherited from the
					# original input regions instead of newly setting it as ''
					label='',
					line=i+1,
				)
			)
	return region_list
