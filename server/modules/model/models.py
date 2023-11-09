from enum import Enum
from typing import List, Optional

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
	label: Optional[str] = ''
	line: Optional[int] = Field(
		0,
		description='Stores the sequential line number of the para text starting from 1'
	)

	def to_xyxy(self) -> tuple[int, int, int, int]:
		return (
			self.bounding_box.x,
			self.bounding_box.y,
			self.bounding_box.x + self.bounding_box.w,
			self.bounding_box.y + self.bounding_box.h
		)

	@classmethod
	def from_xyxy(cls, coords: tuple[int, int, int, int], label='', line=0):
		return cls.from_bounding_box(
			bbox=BoundingBox.from_xyxy(coords),
			label=label,
			line=line
		)

	@classmethod
	def from_bounding_box(cls, bbox, label='', line=0):
		"""
		construct a Region class from the bounding box class
		"""
		return cls(
			bounding_box=bbox,
			label=label,
			line=line,
		)

	@classmethod
	def from_full_table_response(cls, full_table_response):
		"""
        Construct instances of the class from a list of dictionaries containing table responses.
        """
		for response_dict in full_table_response:
			bbox = response_dict.get('bbox', None)
			nrows = response_dict.get('nrows', None)
			ncells = response_dict.get('ncells', None)
			cellrows = response_dict.get('cellrows', None)
        # Create an instance of the class with the provided values.
		return cls(
            bounding_box=bbox,
            nrows=nrows,
            ncells=ncells,
			cellrows=cellrows
        )


class LayoutResponse(BaseModel):
	regions: List[Region]


class LayoutImageResponse(BaseModel):
	"""
	Model class for holding the layout response for one single image
	"""
	image_name: str
	regions: List[Region]