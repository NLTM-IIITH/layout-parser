import sys
sys.path.append(".")
from server.modules.script_identification.iitb_script_identification_model import AlexNet
import requests
from io import BytesIO
from os.path import join
from server.modules.script_identification.models import SIResponse
from torchvision import transforms
import torch
from PIL import Image
import os
import json

def main(folder_path: str) ->list[SIResponse]:
    """Performs inference using AlexNet model

    Args:
        folder_path (str): folder_path of images to perform inference on

    Returns:
        predictions (list[SIResponse]): Prediction in SIResponse format
    """
    files = [join(folder_path, image) for image in os.listdir(folder_path)]
    model = AlexNet(num_classes=11)
    #Load saved weights
    local_weights_path = "iitb-script-identification.pt"
    if os.path.exists(local_weights_path):
    # Load the model from the local file if it exists
        print("Loading model from storage.")
        model.load_state_dict(torch.load(local_weights_path))
    else:
    #If not found locally, download model weights
        print("Downloading model weights")
        weights_url = "https://github.com/kuna71/layout-parser-api/releases/download/weights/Synthetic_wordgenerator_AlexNet_100000_random-font-size_StateDicts.pt"
        response = requests.get(weights_url)
        if response.status_code == 200:
            weights_bytes = BytesIO(response.content)
            model.load_state_dict(torch.load(weights_bytes))
            #Save weights locally
            torch.save(model.state_dict(), local_weights_path)
        else:
            print("Unable to download weights")

    predictions = []    #Stores results
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
    for file in files:
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
    with open(join(folder_path, "output.json"), "w") as f:
        json.dump(predictions, f)

if(__name__ == "__main__"):
    if(len(sys.argv)>1):
        main(sys.argv[1])
    else:
        print("Invalid argument list")
        sys.exit(1)