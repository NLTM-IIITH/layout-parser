from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
	doctr = 'doctr'
	craft = 'craft'
	v2_doctr = 'v2_doctr'
	worddetector = 'worddetector'
	v2_docTR_readingOrder = 'v2_docTR_readingOrder'
	v1_textAttb = 'v1_textAttb'
	v1_urdu = 'v1_urdu'
	tesseract = 'tesseract'
	cropPadFix = 'cropPadFix'


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
	order: Optional[int] = -1
	label: Optional[str] = ''
	line: Optional[int] = Field(
		0,
		description='Stores the sequential line number of the para text starting from 1'
	)
	attributes: Optional[dict]= {}

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
	def from_bounding_box(cls, bbox,attb={},order=-1,label='', line=0):
		"""
		construct a Region class from the bounding box class
		"""
		return cls(
			bounding_box=bbox,
			order=order,
			label=label,
			line=line,
			attributes = attb
		)


class LayoutResponse(BaseModel):
	regions: List[Region]


class LayoutImageResponse(BaseModel):
	"""
	Model class for holding the layout response for one single image
	"""
	image_name: str
	regions: List[Region]
