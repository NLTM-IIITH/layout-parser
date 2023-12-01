# routes.py
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from server.modules.layout_detection.helpers import get_layout_from_single_image


router = APIRouter(
	prefix='/layout/detect-layouts',
	tags=['Layout-Detection'],
)

@router.post("/")
async def detect_layout(file: UploadFile):
    """
	API endpoint for detecting the layouts given an input image
	"""
    try:
        # Save the uploaded image temporarily
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        # Process the uploaded image
        layout = get_layout_from_single_image(file.filename)
        # Return the layout data as a JSON response
        return JSONResponse(content={"message":"Layout Detection Successfull","layout": layout})
    except Exception as e:
        return JSONResponse(content={"message":"Layout Detection Failed","error": str(e)}, status_code=500)
