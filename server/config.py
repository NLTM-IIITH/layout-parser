from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    image_folder: str = '/home/layout/layout-parser/images'
    tesseract_language: dict[str, str] = {
        'english': 'eng',
        'hindi': 'hin',
        'marathi': 'mar',
        'tamil': 'tam',
        'telugu': 'tel',
        'kannada': 'kan',
        'gujarati': 'guj',
        'punjabi': 'pan',
        'bengali': 'ben',
        'malayalam': 'mal',
        'assamese': 'asm',
        'manipuri': 'ben',
        'oriya': 'ori',
        'urdu': 'urd',
        # Minor Languages
        'bodo': 'hin',
        'dogri': 'hin',
        'kashmiri': 'hin',
        'konkani': 'hin',
        'maithili': 'ben',
        'nepali': 'nep',
        'santali': 'hin',
        'sindhi': 'snd',
        'sanskrit': 'san',
    }
    yolov1_model_path: Path = Path('/home/layout/models/yolo_v1')
    yolov2_model_path: Path = Path('/home/layout/models/yolo_v2')
    v03xx_model_path: Path = Path('/home/layout/models/urdu_line')
    v04xx_model_path: Path = Path('/home/layout/models/ajoy_word_segmentation')

settings = Settings()