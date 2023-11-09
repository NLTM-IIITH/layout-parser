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
from .table_cellwise_detection import *
from .ocr_config import *

# TODO: remove this line and try to set the env from the docker-compose file.
os.environ['USE_TORCH'] = '1'

def logtime(t: float, msg:  str) -> None:
	print(f'[{int(time.time() - t)}s]\t {msg}')

t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_multiple_image_fasterrcnn(folder_path: str) -> List[LayoutImageResponse]: 
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the word detected regions.

	@returns list of BoundingBox class
	"""
	img_files = [join(folder_path, i) for i in os.listdir(folder_path)]
	
	ret = []
	for idx in range(len(img_files)):
		t = time.time()
		full_table_response = get_tables_from_page(img_files[idx])
		logtime(t, 'Time taken to perform fastcrnn inference')
		regions = Region.from_full_table_response(full_table_response)
		ret.append(
			LayoutImageResponse(
				regions=regions.copy(),
			    image_name=os.path.basename(img_files[idx])
				)
			)
	logtime(t, 'Time taken to process the doctr output')
	return ret

def process_image_fasterrcnn(image_path: str, model: str='fasterrcnn') -> List[Region]:
	"""
	given the path of the image, this function returns a list
	of bounding boxes of all the table detected regions.

	@returns list of BoundingBox class
	"""
	print('performing table detection')
	
	ret = []
	
	t = time.time()
	full_table_response = get_tables_from_page(image_path)
	logtime(t, 'Time taken to perform fastcrnn inference')
	regions = Region.from_full_table_response(full_table_response)
	ret = LayoutImageResponse(
				regions=regions.copy(),
			    image_name=os.path.basename(image_path)
				)
	logtime(t, 'Time taken to process the fastcrnn output')
	return ret

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