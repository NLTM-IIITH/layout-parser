from os.path import join

import cv2
from tqdm import tqdm

from ..core.config import IMAGE_FOLDER
from .models import LayoutImageResponse, Region


def process_region(img, region: Region) -> Region:
	line = region.line
	label = region.label
	x1, y1, x2, y2 = region.to_xyxy()

	# process left
	while img[y1:y2, x1].sum() > 0:
		x1 -= 1
	
	# process top
	while img[y1, x1:x2].sum() > 0:
		y1 -= 1
	
	# process right
	while img[y1:y2, x2].sum() > 0:
		x2 += 1
	
	# process bottom
	while img[y2, x1:x2].sum() > 0:
		y2 += 1

	return Region.from_xyxy(
		(x1, y1, x2, y2),
		line=line,
		label=label
	)


def process_dilate(regions: list[Region], image_path: str) -> list[Region]:
	img = cv2.imread(image_path, 0)
	_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
	ret = []
	for region in tqdm(regions):
		ret.append(process_region(img, region))
	return ret

def process_multiple_dilate(inp: list[LayoutImageResponse]) -> list[LayoutImageResponse]:
	ret = []
	for i in inp:
		ret.append(
			LayoutImageResponse(
				image_name=i.image_name,
				regions=process_dilate(i.regions, join(IMAGE_FOLDER, i.image_name))
			)
		)
	return ret