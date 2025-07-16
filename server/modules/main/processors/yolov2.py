from glob import glob
from os.path import basename, join
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from server.config import settings

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


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

def get_model_predict(model: YOLO, input_image: str, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
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
    predictions = transform_predict_to_df(predictions, model.model.names) # type: ignore
    return predictions

class YoloV2Processor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        language = kwargs.get('language', 'hindi')
        model_path = settings.yolov2_model_path / f'{language}.pt'
        if not model_path.exists():
            model_path = settings.yolov2_model_path / 'auto.pt'
        model = YOLO(str(model_path))
        ret = []
        for image in tqdm(glob(join(folder_path, '*')), desc='Performing inference'):
            print(image)
            result = get_model_predict(
                model=model,
                input_image=image,
                conf=0.15,
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

register(ModelChoice.yolov2)(YoloV2Processor())