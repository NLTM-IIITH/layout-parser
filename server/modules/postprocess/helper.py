import base64
import json
import os
import torch
from torchvision import transforms
from PIL import Image
from os.path import join

from fastapi import HTTPException

from .models import ClassifyResponse, SIResponse
from Models.script_identification import AlexNet
import requests
from io import BytesIO
def script_inference_alexnet(folder_path: str) ->list[SIResponse]:
	"""Performs inference using AlexNet model

	Args:
		folder_path (str): folder_path of images to perform inference on

	Returns:
		predictions (list[SIResponse]): Prediction in SIResponse format
	"""
	files = [join(folder_path, image) for image in os.listdir(folder_path)]
	weights_url = ""
	response = requests.get(weights_url)
	if response.status_code == 200:
    # Load the weights from the response content
		weights_bytes = BytesIO(response.content)
	else:
		print("Unable to download weights")
	predictions = []
	for file in files:
		model = AlexNet(num_classes=11)
    	#Load saved weights
		model.load_state_dict(torch.load(weights_bytes))
		#Define transformation
		normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		)
		preprocess = transforms.Compose([
			transforms.Resize((227,227)),
			transforms.ToTensor(),
			normalize,
			transforms.Grayscale()
		])
		scripts = ["devanagari", "bengali", "gujarati", "gurumukhi", "kannada", "malayalam", "odia", "tamil", "urdu", "latin", "odia"]
  

		#Load image and apply transformations
		image = Image.open(file).convert("RGB")
		input_tensor = preprocess(image)
		input_tensor = input_tensor.unsqueeze(0)  
		#Get output
		with torch.no_grad():  
			output = model(input_tensor)
		probabilities = torch.softmax(output, dim=1)
		predicted_class_index = torch.argmax(probabilities, dim=1)
		predicted_class = scripts[predicted_class_index.item()]
		predictions.append(SIResponse(text=predicted_class))
	return predictions 

def process_images(images: list[str], path: str='/home/layout/layout-parser/images'):
	"""
	processes all the images in the given list.
	it saves all the images in the /home/ocr/website/images folder and
	returns this absolute path.
	"""
	print('deleting all the previous data from the images folder')
	os.system(f'rm -rf {path}/*')
	for idx, image in enumerate(images):
		if image is not None:
			try:
				# saving the base64 image as JPEG
				assert isinstance(image, str)
				with open(join(path, f'{idx}.jpg'), 'wb') as f:
					f.write(base64.b64decode(image))
			except:
				raise HTTPException(
					status_code=400,
					detail=f'Error while decoding and saving the image #{idx}',
				)
		else:
			raise HTTPException(
				status_code=400,
				detail=f'image #{idx} doesnt contain either imageContent or imageUri',
			)


def process_layout_output(
	path: str = '/home/layout/layout-parser/images'
) -> list[ClassifyResponse]:
	"""
	process the ./images/out.json file and returns the ocr response.
	"""
	try:
		a = open(join(path, 'out.json'), 'r').read().strip()
		a = json.loads(a)
		a = list(a.items())
		a = sorted(a, key=lambda x:int(x[0].split('.')[0]))
		return [ClassifyResponse(text=i[1]) for i in a]
	except Exception as e:
		print(e)
		raise HTTPException(
			status_code=500,
			detail='Error while parsing the ocr output'
		)