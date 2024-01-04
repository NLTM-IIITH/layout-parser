from pydantic import BaseModel
from fastapi import UploadFile

class LayoutDetection(BaseModel):
    """
    Pydantic model representing an uploaded file for layout detection.
    Attributes:
        file (UploadFile): The uploaded file to be processed for layout detection.
    """
    file: UploadFile
