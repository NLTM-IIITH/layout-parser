from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
	doctr = 'doctr'
	craft = 'craft'
	v2_doctr = 'v2_doctr'


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


class Region(BaseModel):
	bounding_box: BoundingBox
	label: Optional[str] = ''
	line: Optional[int] = Field(
		0,
		description='Stores the sequential line number of the para text starting from 1'
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


class LayoutResponse(BaseModel):
	regions: List[Region]


class LayoutImageResponse(BaseModel):
	"""
	Model class for holding the layout response for one single image
	"""
	image_name: str
	regions: List[Region]
