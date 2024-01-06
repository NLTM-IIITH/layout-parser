import os
import shutil
import cv2
from server.modules.layout_detection.classes.equations import get_equation_detection
from server.modules.layout_detection.classes.figures import get_figure_detection
from server.modules.layout_detection.classes.tables import get_tables_cells_detection
from server.modules.layout_detection.classes.utilities import mask_image
from typing import List, Tuple
from os.path import join
from fastapi import UploadFile

def get_layout_from_single_image(image_name):
    layout = {}
    image = cv2.imread(image_name)
    height, width, _ = image.shape
    table_bboxes, cell_bboxes = get_tables_cells_detection(image_name)
    masked_image = mask_image(image, table_bboxes)
    equation_bboxes = get_equation_detection(masked_image)
    masked_image = mask_image(image, equation_bboxes)
    figure_bboxes = get_figure_detection(masked_image)
    layout["image-name"] = image_name
    layout["height"] = height
    layout["width"] = width
    layout["tables"] = table_bboxes
    layout["cells"] = cell_bboxes
    layout['equations'] = equation_bboxes
    layout["figures"] = figure_bboxes
    #layout["masked-image"] = masked_image.tolist()

    #converting the numpy arrays to python lists
    layout["tables"] = [table.tolist() for table in layout["tables"]]
    layout["cells"] = [cell.tolist() for cell in layout["cells"]]
    layout["equations"] = [equation.tolist() if not isinstance(equation, list) else equation for equation in layout["equations"]]
    layout["figures"] = [figure.tolist() if not isinstance(figure, list) else figure for figure in layout["figures"]]


    return layout


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