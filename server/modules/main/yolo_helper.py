import glob
from os.path import basename, exists, join

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from .models import BoundingBox, LayoutImageResponse, Region
from .yolo_ro_classes import BoundingBoxVisualizer, YOLOJsonExtractor


def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
        imgsz=image_size, 
        source=input_image, 
        conf=conf,
        save=save, 
        augment=augment,
        flipud= 0.0,
        fliplr= 0.0,
        mosaic = 0.0,
        device = [0 if torch.cuda.is_available() else "cpu"]
    )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions

def do_yolo_infer_v1(path, language='hindi', nms_threshold: float = 0.15):
    model_path = f'/home/layout/models/yolo_v1/{language}.pt'
    model = YOLO(model_path)
    ret = []
    print(path)
    for image in tqdm(glob.glob(join(path, '*')), desc='Performing inference'):
        print(image)
        result = get_model_predict(
            model=model,
            input_image=image,
            conf=nms_threshold,
            augment=False,
            image_size=1280,
            save=False
        )
        regions = []
        for i in range(len(result)):
            regions.append(
                Region.from_bounding_box(
                    BoundingBox.from_xyxy((
                        int(result['xmin'][i]),
                        int(result['ymin'][i]),
                        int(result['xmax'][i]),
                        int(result['ymax'][i]),
                    ))
                )
            )
        ret.append(LayoutImageResponse(
            image_name=basename(image),
            regions=regions.copy(),
        ))
    return ret



def do_yolo_infer_v2(path, language='hindi', nms_threshold: float = 0.15):
    model_path = f'/home/layout/models/yolo_v2/{language}.pt'
    if not exists(model_path):
        model_path = f'/home/layout/models/yolo_v2/auto.pt'
    model = YOLO(model_path)
    ret = []
    print(path)
    for image in tqdm(glob.glob(join(path, '*')), desc='Performing inference'):
        print(image)
        result = get_model_predict(
            model=model,
            input_image=image,
            conf=nms_threshold,
            augment=False,
            image_size=1280,
            save=False
        )
        regions = []
        for i in range(len(result)):
            regions.append(
                Region.from_bounding_box(
                    BoundingBox.from_xyxy((
                        int(result['xmin'][i]),
                        int(result['ymin'][i]),
                        int(result['xmax'][i]),
                        int(result['ymax'][i]),
                    ))
                )
            )
        ret.append(LayoutImageResponse(
            image_name=basename(image),
            regions=regions.copy(),
        ))
    return ret


def do_yolo_infer_ro(path, language='hindi', nms_threshold: float = 0.15):
    model_path = f'/home/layout/models/yolo_v2/{language}.pt'
    if not exists(model_path):
        model_path = f'/home/layout/models/yolo_v2/auto.pt'
    yolo_extractor = YOLOJsonExtractor(model_path)
    visualizer = BoundingBoxVisualizer(delta_y=50)
    ret = []
    print(path)
    for image in tqdm(glob.glob(join(path, '*')), desc='Performing inference'):
        # 1 is the pagenumber for the bboxes, which is static.
        result_json = yolo_extractor.process_image(image, 1)
        ordered_boxes = visualizer.get_reading_order(result_json)
        regions = []
        for lineno, line in enumerate(ordered_boxes):
            for _, word in enumerate(line):
                regions.append(
                    Region.from_bounding_box(
                        BoundingBox.from_xyxy((
                            word['bounding_box']['x_min'],
                            word['bounding_box']['y_min'],
                            word['bounding_box']['x_max'],
                            word['bounding_box']['y_max'],
                        )),
                        line=lineno + 1,
                        order=word['reading_order']
                    )
                )
        ret.append(LayoutImageResponse(
            image_name=basename(image),
            regions=regions.copy(),
        ))
    return ret