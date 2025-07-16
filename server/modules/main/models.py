from enum import Enum

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    doctr = 'doctr'
    craft = 'craft'
    v2_doctr = 'v2_doctr'
    worddetector = 'worddetector'
    v1_urdu = 'v1_urdu'
    yolov1 = 'yolo_v1'
    yolov2 = 'yolo_v2'
    textron = 'textron'
    hisam = 'hisam'
    openseg = 'openseg'
    yoloro = 'yolo_ro'
    textbpnpp = 'textbpnpp'
    v0301 = 'V-03.01'
    v0302 = 'V-03.02'
    v0303 = 'V-03.03'
    v0304 = 'V-03.04'
    v0401 = 'V-04.01'
    v0402 = 'V-04.02'
    v0403 = 'V-04.03'
    v0404 = 'V-04.04'
    v0405 = 'V-04.05'
    v0501 = 'V-05.01' # Merge V-04.02 + Openseg
    v0502 = 'V-05.02' # Merge V-04.02 + Openseg + Craft
    # merge3 = 'merge3'


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

    def __eq__(self, other: "BoundingBox") -> bool:
        center = (self.x+self.w/2, self.y+self.h/2)
        other_center = (other.x+other.w/2, other.y+other.h/2)
        return all((
            other.x <= center[0] <= other.x + other.w,
            other.y <= center[1] <= other.y + other.h,
            self.x <= other_center[0] <= self.x + self.w,
            self.y <= other_center[1] <= self.y + self.h,
        ))

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.x + self.w // 2,
            self.y + self.h // 2
        )

    def overlaps(self, other: "BoundingBox") -> bool:
        """check if the 2 bboxes have any overlap"""

        return not (
            self.x + self.w <= other.x or  # self is left of other
            other.x + other.w <= self.x or  # other is left of self
            self.y + self.h <= other.y or  # self is above other
            other.y + other.h <= self.y     # other is above self
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
    order: int = -1
    label: str = ''
    line: int = Field(
        0,
        description='Stores the sequential line number of the para text starting from 1'
    )
    confidence: float = 0.0
    attributes: dict = {}

    def __eq__(self, other: "Region"):
        return self.bounding_box == other.bounding_box

    def __hash__(self):
        return hash((
            self.bounding_box.x,
            self.bounding_box.y,
            self.bounding_box.w,
            self.bounding_box.h
        ))

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (
            self.bounding_box.x,
            self.bounding_box.y,
            self.bounding_box.x + self.bounding_box.w,
            self.bounding_box.y + self.bounding_box.h
        )

    @classmethod
    def from_xyxy(cls, coords: tuple[int, int, int, int], label='', line=0, conf=0.0, order=-1):
        return cls.from_bounding_box(
            bbox=BoundingBox.from_xyxy(coords),
            label=label,
            line=line,
            confidence=conf,
            order=order,
        )

    @classmethod
    def from_bounding_box(cls, bbox,attb={},order=-1,label='', line=0, confidence=0.0):
        """
        construct a Region class from the bounding box class
        """
        return cls(
            bounding_box=bbox,
            order=order,
            label=label,
            line=line,
            confidence=confidence,
            attributes = attb
        )


class LayoutResponse(BaseModel):
    regions: list[Region]


class LayoutImageResponse(BaseModel):
    """
    Model class for holding the layout response for one single image
    """
    image_name: str
    regions: list[Region]