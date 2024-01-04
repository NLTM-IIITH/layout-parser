import numpy as np
import cv2
import torch
import glob as glob
import torchvision
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytesseract
import requests

from .ocr_config import *


# Returns Faster RCNN model to perfrom table cell-wise detection
def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

# Returns true if two rectangles b1 and b2 overlap, b is of the form [x1, y1, x2, y2]
def do_overlap(b1, b2):
    if (b1[0] >= b2[2]) or (b1[2] <= b2[0]) or (b1[3] <= b2[1]) or (b1[1] >= b2[3]):
         return False
    else:
        return True

# Get cells which are a part of a table
def get_cells_from_table(tab, cells):
    tablecells = []
    for c in cells:
        overlap = do_overlap(tab, c)
        if overlap:
            tablecells.append(c)
    return tablecells

def perform_nms(boxes, scores, nms_threshold):
    dets = torch.Tensor(boxes)
    scores = torch.Tensor(scores)
    res = nms(dets, scores, nms_threshold)
    final_boxes =[]
    for ind in res:
        final_boxes.append(boxes[ind])
    return final_boxes

def get_tables_from_page(img_file):
    full_table_response = []
    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=3).to(device)
    response = requests.get(faster_rcnn_model_url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the downloaded content to a file (e.g., model.pth)
        with open('table_detection_model.pth', 'wb') as model_file:
            model_file.write(response.content)

        # Load the model.pth file using torch.load
        model.load_state_dict(torch.load('table_detection_model.pth', map_location=device))

    else:
        print("Failed to download the model file")
    model.eval()

    # classes: 0 index is reserved for background
    CLASSES = [
        'bkg', 'table', 'cell'
    ]
    # any detection having score below this will be discarded
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

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        print('Table Prediction Complete')

        # Trim classes for top k boxes predicted over threshold score
        classes = pred_classes[:len(boxes)]

        # Collect table and cells 
        unfiltered_tables = []
        unfiltered_cells = []
        # Collect Scores
        table_scores = []
        cell_scores = []
        for i in range(len(boxes)):
            if classes[i] == 'table':
                unfiltered_tables.append(boxes[i])
                table_scores.append(scores[i])
            else:
                unfiltered_cells.append(boxes[i])
                cell_scores.append(scores[i])

        tables = []
        cells = []
        # Perform NMS to resolve overlap issue
        if len(unfiltered_tables):
            tables = perform_nms(unfiltered_tables, table_scores, nms_table_threshold)
        if len(unfiltered_cells):
            cells = perform_nms(unfiltered_cells, cell_scores, nms_cell_threshold)

        for tablebbox in tables:
        
            tabcells = get_cells_from_table(tablebbox, cells)
            # print(tablebbox)
            # print(len(tabcells))

            # Proceed only if table has cells
            if len(tabcells) > 0:
                # Sort cells based on y coordinates
                strcells = sorted(tabcells, key=lambda b:b[1]+b[3], reverse=False)

                # Calculate Mean height
                cell_heights = [c[3] - c[1] for c in tabcells]
                mean_height = int(np.mean(cell_heights))

                # Assign row to each cell based on y coordinate wise arrangement 
                cellrow = [0]
                assign_row = 0
                for i in range(len(strcells) - 1):
                    consec_cell_height = strcells[i + 1][1] - strcells[i][1]
                    if consec_cell_height > row_determining_threshold * mean_height:
                        assign_row = assign_row + 1
                    cellrow.append(assign_row)

                # Get number of rows and columns
                rows = list(set(cellrow))
                nrows = len(rows)
                counts = [0] * nrows
                for cr in cellrow:
                    counts[cr] = counts[cr] + 1
                ncols = max(counts)

                # Generate row-wise cell sequences of bounding boxes
                cellrows = {}
                for i in rows:
                    row_wise_cells = []
                    for j in range(len(strcells)):
                        if i == cellrow[j]:
                            row_wise_cells.append(strcells[j])
                    row_wise_cells = sorted(row_wise_cells, key=lambda b:b[0], reverse=False)
                    cellrows[i] = row_wise_cells

                tableresponse = {}
                tableresponse['bbox'] = tablebbox 
                tableresponse['nrows'] = nrows
                tableresponse['ncells'] = len(strcells)
                tableresponse['cellrows'] = cellrows
                full_table_response.append(tableresponse)
    return full_table_response