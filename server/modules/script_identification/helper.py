import json
from os.path import join
import base64
import json
import os
import torch
from torchvision import transforms
from PIL import Image
from os.path import join

from fastapi import HTTPException

from .models import SIResponse
from .iitb_script_identification_model import AlexNet

def script_inference_alexnet(folder_path: str) ->list[SIResponse]:
    """Performs inference using AlexNet model

    Args:
        folder_path (str): folder_path of images to perform inference on
    """
    files = [join(folder_path, image) for image in os.listdir(folder_path)]
    predictions = []
    for file in files:
        model = AlexNet(num_classes=11)
        #Load saved weights
        model.load_state_dict(torch.load("script-identification-iitb.pth"))
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
        predictions.append(predicted_class)
    #Save output in output.json
    with open("data/output.json", "w") as f:
        json.dump(predictions, f)

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
def process_output(path: str = "server/modules/script_identification/output.json"):
    """Processes output.json and returns in response format

    Args:
        path (str, optional): Path to output.json. Defaults to "server/modules/script_identification/output.json".

    Returns:
        List[SIResponse]: Processed output
    """
    try:
        with open(join(path, "output.json"), 'r') as json_file:
            loaded=json.load(json_file)
            print(loaded)
            ret = [SIResponse(text=i) for i in loaded]
            print(ret)
            return ret
    except:
        print("Error while trying to open output file")