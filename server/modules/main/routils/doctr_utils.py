from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection
from doctr.utils.visualization import visualize_page
import json
import os

def doctr_predictions(directory, predictor):
#     #Gets the predictions from the model
    
    doc = DocumentFile.from_images(directory)
    result = predictor(doc)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    abs_coords = [
    [[int(round(word['geometry'][0][0] * dims[1])), 
      int(round(word['geometry'][0][1] * dims[0])), 
      int(round(word['geometry'][1][0] * dims[1])), 
      int(round(word['geometry'][1][1] * dims[0]))] for word in words]
    for words, dims in zip(regions, page_dims)
    ]

    return abs_coords

def get_doctr_preds_from_json(json_file, rel_dir):
    with open(json_file) as f:
        data = json.load(f)
    print(data.keys())
    for key in data.keys():
        if rel_dir == key:
            # return data[key]
            # return list(map(int, data[key]))
            intt =  [[int(element) for element in sublist] for sublist in data[key]]
            return [[[sublist[2], sublist[0], sublist[3], sublist[1]] for sublist in intt]]
        
def get_doctr_preds_from_json_2(json_file, rel_dir):
    with open(json_file) as f:
        data = json.load(f)
    print(data.keys())
    for key in data.keys():
        if rel_dir == key:
            # return data[key]
            # return list(map(int, data[key]))
            intt =  [[int(element) for element in sublist] for sublist in data[key][0]]
            return [intt]
    