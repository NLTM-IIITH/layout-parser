from glob import glob
from os.path import basename, join
from typing import Any

from tqdm import tqdm
from ultralytics import YOLO

from server.config import settings

from ..factory import register
from ..models import BoundingBox, LayoutImageResponse, ModelChoice, Region


class YOLOJsonExtractor:
    """
    A class to handle YOLO object detection and return bounding boxes in JSON format.
    """
    def __init__(self, model_path):
        """
        Initialize the YOLOJsonExtractor with a specified YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)

    def get_boxes_as_json(self, result, page_number):
        """
        Convert YOLO bounding boxes to JSON format without text or reading order.

        Args:
            result: YOLO model result object containing boxes.
            page_number (int): The page number corresponding to the detection.

        Returns:
            dict: JSON data with bounding boxes.
        """
        boxes = result.boxes  # Assuming this is a YOLOv8 result with a Boxes object
        xyxy_boxes = boxes.xyxy.cpu().numpy()  # Convert to numpy array (detach if on GPU)

        words = []
        for box in xyxy_boxes:
            x_min, y_min, x_max, y_max = box
            word_entry = {
                "bounding_box": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max)
                }
            }
            words.append(word_entry)

        output_data = {
            "page": page_number,
            "words": words
        }
        return output_data

    def process_image(self, image_path, page_number, conf=0.15, iou=0.15, img_size=(1024, 1024)):
        """
        Process an input image, run inference using YOLO, and return the results as JSON.

        Args:
            image_path (str): Path to the input image.
            page_number (int): Page number for the output JSON.
            conf (float): Confidence threshold for YOLO detection.
            iou (float): IoU threshold for YOLO detection.
            img_size (tuple): Image size to resize input images.

        Returns:
            dict: JSON data with bounding boxes for the image.
        """
        results = self.model(
            source=image_path,
            conf=conf,
            iou=iou,
            save_txt=False,
            save=False,
            imgsz=img_size,
            verbose=False
        )

        # Use the first result for processing (assuming single image input)
        result = results[0]
        return self.get_boxes_as_json(result, page_number)


class BoundingBoxVisualizer:
    """
    A class to visualize bounding boxes, assign reading order, and draw them on an image.
    """

    def __init__(self, delta_y=50):
        """
        Initialize the BoundingBoxVisualizer with parameters for grouping rows.

        Args:
            delta_y (int): Threshold for grouping bounding boxes into rows based on vertical proximity.
        """
        self.delta_y = delta_y

    def get_reading_order(self, bounding_boxes):
        """
        Assign reading order to bounding boxes.

        Args:
            bounding_boxes (dict): JSON data with bounding boxes.

        Returns:
            list: List of bounding boxes with `reading_order` assigned.
        """
        # Step 1: Sort bounding boxes by `y_min` to group by rows
        bounding_boxes = sorted(bounding_boxes["words"], key=lambda box: box["bounding_box"]['y_min'])
        rows = []
        current_row = [bounding_boxes[0]]

        # Step 2: Group boxes into rows based on `delta_y` threshold
        for box in bounding_boxes[1:]:
            if abs(box["bounding_box"]['y_min'] - current_row[-1]["bounding_box"]['y_min']) < self.delta_y:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)  # Add the last row

        # Step 3: Sort each row by `x_min` (left to right)
        
        ordered_rows = []
        reading_order = 1
        for row in rows:
            ordered_boxes = []
            row_sorted = sorted(row, key=lambda box: box["bounding_box"]['x_min'])
            for box in row_sorted:
                box['reading_order'] = reading_order
                ordered_boxes.append(box)
                reading_order += 1
            ordered_rows.append(ordered_boxes)
        return ordered_rows


class YoloROProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        language = kwargs.get('language', 'hindi')
        model_path = settings.yolov2_model_path / f'{language}.pt'
        if not model_path.exists():
            model_path = settings.yolov2_model_path / 'auto.pt'
        yolo_extractor = YOLOJsonExtractor(model_path)
        visualizer = BoundingBoxVisualizer(delta_y=50)
        ret = []
        for image in tqdm(glob(join(folder_path, '*')), desc='Performing inference'):
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

register(ModelChoice.yoloro)(YoloROProcessor())