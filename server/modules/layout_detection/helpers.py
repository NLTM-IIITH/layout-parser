import os
import shutil

from typing import List
from os.path import join
from fastapi import UploadFile


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