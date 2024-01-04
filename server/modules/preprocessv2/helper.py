"""
Helper functions for preprocessing

Author: Vignesh E (vignesh1234can@gmail.com)
"""

import os
import shutil
from os.path import join
from typing import List, Tuple

from fastapi import UploadFile
from .models import  BoundingBox

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def save_uploaded_images(images: List[UploadFile],image_dir) -> str:
	print('removing all the previous uploaded files from the image folder')
	delete_files_in_directory(image_dir)
	print(f'Saving {len(images)} to location: {image_dir}')
	for image in images:
		location = join(image_dir, f'{image.filename}')
		with open(location, 'wb') as f:
			shutil.copyfileobj(image.file, f)
	return image_dir

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
