import numpy as np
import os
import cv2
import requests
import torch
import torchvision
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .config import *


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def perform_nms(boxes, scores, nms_threshold):
    dets = torch.Tensor(boxes)
    scores = torch.Tensor(scores)
    res = nms(dets, scores, nms_threshold)
    final_boxes = []
    for ind in res:
        final_boxes.append(boxes[ind])
    return final_boxes

def get_tables_cells_detection(img_file):
    # set the computation device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(num_classes=3).to(device)
    model_file_path = 'table_detection_model.pth'

    # Check if the model file already exists
    if os.path.exists(model_file_path):
        # Load the model from the local file
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        print("Model loaded from the local file.")
    else:
        # Download the model if it doesn't exist locally
        response = requests.get(faster_rcnn_model_url)
    
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the downloaded content to a file (e.g., model.pth)
            with open(model_file_path, 'wb') as model_file:
                model_file.write(response.content)

            # Load the model.pth file using torch.load
            model.load_state_dict(torch.load(model_file_path, map_location=device))
            print("Model downloaded and loaded.")
        else:
            print("Failed to download the model file")

    model.eval()

    # classes: 0 index is reserved for background
    CLASSES = ["bkg", "table", "cell"]
    # any classes having score below this will be discarded
    detection_threshold = det_threshold

    # get the image file name for saving output later on
    image_name = img_file
    image = cv2.imread(image_name)
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all classes to CPU for further operations
    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes

    # Final lists to return
    tables = []
    cells = []

    if len(outputs[0]["boxes"]) != 0:
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]
        print("Table Prediction Complete")

        # Trim classes for top k boxes predicted over threshold score
        classes = pred_classes[: len(boxes)]

        # Collect table and cells
        unfiltered_tables = []
        unfiltered_cells = []
        # Collect Scores
        table_scores = []
        cell_scores = []
        for i in range(len(boxes)):
            if classes[i] == "table":
                unfiltered_tables.append(boxes[i])
                table_scores.append(scores[i])
            else:
                unfiltered_cells.append(boxes[i])
                cell_scores.append(scores[i])

        # Perform NMS to resolve overlap issue
        if len(unfiltered_tables):
            tables = perform_nms(unfiltered_tables, table_scores, nms_table_threshold)
        if len(unfiltered_cells):
            cells = perform_nms(unfiltered_cells, cell_scores, nms_cell_threshold)

    return tables, cells