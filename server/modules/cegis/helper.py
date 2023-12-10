import base64
import csv
import imghdr
import os
import shutil
from os.path import basename, join
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

from .models import *

# TODO: remove this line and try to set the env from the docker-compose file.
os.environ['USE_TORCH'] = '1'


def process_image_url(image_url: str, savepath: str, resize=None) -> None:
	"""
	input the url of the image and download and saves the image inside the folder.
	savename is the name of the image to be saved as
	"""
	tmp = TemporaryDirectory(prefix='save_image')
	r = requests.get(image_url, stream=True, verify=False)
	print(r)
	if r.status_code == 200:
		r.raw.decode_content = True
		img_path = join(tmp.name, 'image')
		with open(img_path, 'wb') as f:
			shutil.copyfileobj(r.raw, f)
		img = Image.open(img_path)
		img = ImageOps.exif_transpose(img)
		if imghdr.what(img_path) == 'png':
			print('image is PNG format, converting to JPG')
			img = img.convert('RGB')
		if resize:
			img = img.resize(resize)
		img.save(savepath)
		print('downloaded the image:', image_url)
		return (img.width, img.height)
	else:
		raise Exception('status_code is not 200 while downloading the image from url')


def save_image(url: str, dir_path: str, template_size) -> str:
	ret = join(dir_path, 'image.jpg')
	process_image_url(url, ret, template_size)
	return ret


def save_template_image(url: str, dir_path: str) -> str:
	ret = join(dir_path, 'template.jpg')
	tr = process_image_url(url, ret)
	return (ret, tr)

def save_template_coords(url: str, dir_path: str) -> str:
	ret = join(dir_path, 'template.csv')
	r = requests.get(url, stream=True, verify=False)
	r.raw.decode_content = True
	with open(ret, 'wb') as f:
		shutil.copyfileobj(r.raw, f)
	return ret


def extractImage(img, coordinate_path, saved_images_path):
	count = 1

	with open(coordinate_path, mode = "r") as file:
		next(file)
		csvFile = csv.reader(file)

		for lines in csvFile:
			# print(lines)
			field = str(lines[0]).strip()
			x = int(float(lines[1]))
			y = int(float(lines[3]))
			w = int(float(lines[2])) - x
			h = int(float(lines[4])) - y

			if w == 0 or h == 0:
				print(lines)
				print(x,y,w,h)
				print('Skipping the image at count: {}'.format(count))
	
			else:
				images_name = saved_images_path + "/" + field + '-' + str(count)+ ".jpg"
				single_image = img[y: y+h, x: x+w]
				cv2.imwrite(images_name, single_image)

				count += 1
      

def perform_align(imgPath, saved_images_path, img_template_path, coordinate_path):
	im1 = cv2.imread(img_template_path)
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

	im2 = cv2.imread(imgPath, cv2.IMREAD_COLOR)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

	im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	max_num_features = 500
	orb = cv2.ORB_create(max_num_features)
	keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

	matcher = cv2.DescriptorMatcher_create(
	    cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	matches = sorted(matches, key=lambda x: x.distance, reverse=False)

	numGoodMatches = int(len(matches)*0.1)
	matches = matches[:numGoodMatches]

	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	#Find homography
	h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

	height, width, _ = im1.shape
	im2_reg = cv2.warpPerspective(im2, h, (width, height))
	print(type(im2_reg))
	print(im2_reg.shape)
	# cv2.imwrite(im2_reg[:,:], '/home/layout/image.jpg')
	extractImage(im2_reg, coordinate_path, saved_images_path)

def get_all_images(path):
	a = os.listdir(path)
	a = sorted(a, key=lambda x:int(x.strip().split('.')[0].split('-')[-1]))
	ret = {}
	a = [join(path, i) for i in a]
	for i in a:
		name = basename(i).strip().split('-')[0].strip()
		print(i,name)
		if name in ret:
			ret[name]['images'].append(
				base64.b64encode(open(i, 'rb').read()).decode()
			)
		else:
			ret[name] = {
				'images':[
					base64.b64encode(open(i, 'rb').read()).decode()
				]
			}
	return ret
