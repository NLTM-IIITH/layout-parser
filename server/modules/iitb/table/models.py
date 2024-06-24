from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    fasterrcnn = 'fasterrcnn'


class BoundingBox(BaseModel):
    x: int = Field(
        description='X coordinate of the upper left point of bbox'
    )
    y: int = Field(
        description='Y coordinate of the upper left point of bbox'
    )
    w: int = Field(
        description='width of the bbox (in pixel)'
    )
    h: int = Field(
        description='height of the bbox (in pixel)'
    )

    @classmethod
    def from_xyxy(cls, coords: tuple[int, int, int, int]) -> 'BoundingBox':
        return cls(
            x=coords[0],
            y=coords[1],
            w=coords[2] - coords[0],
            h=coords[3] - coords[1]
        )


class Region(BaseModel):
    bounding_box: BoundingBox
    nrows: Optional[int]
    ncells: Optional[int]
    cellrows: Optional[Dict[int, List[BoundingBox]]]
    
    @classmethod
    def from_full_table_response(cls, full_table_response):
        instances = []  # Create a list to store instances
        for response_dict in full_table_response:
            bbox = response_dict.get('bbox', None)
            nrows = response_dict.get('nrows', None)
            ncells = response_dict.get('ncells', None)
            cellrows_data = response_dict.get('cellrows', None)
            
            if bbox is not None:
                # Extract the x, y, width, and height values from the NumPy array
                coords = bbox
                bbox = BoundingBox.from_xyxy(coords)
            
            cellrows = {}  # Create a dictionary to store cell rows
            if cellrows_data is not None:
                for row_number, bounding_boxes in cellrows_data.items():
                    bounding_boxes = [BoundingBox.from_xyxy(coords) for coords in bounding_boxes]
                    cellrows[int(row_number)] = bounding_boxes
            
            # Create an instance of the class with the provided values and append it to the list
            instances.append(cls(
                bounding_box=bbox,
                nrows=nrows,
                ncells=ncells,
                cellrows=cellrows))
                
        return instances



class LayoutResponse(BaseModel):
    regions: List[Region]


class LayoutImageResponse(BaseModel):
    """
    Model class for holding the layout response for one single image
    """
    image_name: str
    regions: List[Region]