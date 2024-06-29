from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    doctr = 'doctr'
    tesseract = 'tesseract'

class TaskChoice(str, Enum):
    attributes = "font attribute result"
    visualise = "visualisation"

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

    def topolygon(self) -> list[list[int]]:
        """
        converts the {x,y,w,h} type of bounding box to the 4 polygon point bounding
        boxes

        @returns [[x,y], [x,y], [x,y], [x,y]]
        this format of the polygon type of bbox output is compatible with
        the sorting algorithm found on github
        """
        return [
            [self.x, self.y],
            [self.x+self.w, self.y],
            [self.x+self.w, self.y+self.h],
            [self.x, self.y+self.h],
        ]


class FontRegion(BaseModel):
    bounding_box: BoundingBox
    fontSize: Optional[int] = Field(description='size of the font')
    fontColor: Optional[list[int]] = Field(description='font color in RGB')
    fontDecoration: Optional[str] = Field(description='font decoration bold/regular')

class FontAttributeImage(BaseModel):
    image: Optional[str] = Field(description='image name')
    font_regions: list[FontRegion]

class FontAttributesResponse(BaseModel):
    images: list[FontAttributeImage]