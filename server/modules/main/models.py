from enum import Enum
from typing import List, Tuple, Union, Optional

import numpy as np
from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
	doctr = 'doctr'
	craft = 'craft'
	v2_doctr = 'v2_doctr'
	worddetector = 'worddetector'
	textpms = 'textpms'
	dbnet = 'dbnet'


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
	def from_xyxy(cls, coords: Tuple[int, int, int, int]) -> 'BoundingBox':
		return cls(
			x=coords[0],
			y=coords[1],
			w=coords[2] - coords[0],
			h=coords[3] - coords[1]
		)
	
	def to_polygon(self) -> 'List[Point]':
		return [
			Point(self.x, self.y),
			Point(self.x + self.w, self.y),
			Point(self.x + self.w, self.y + self.h),
			Point(self.x, self.y + self.h),
		]


class Point(BaseModel):
	x: int
	y: int

	def __init__(self, x: int, y: int):
		return super().__init__(x=x, y=y)

class PolygonRegion(BaseModel):
	points: List[Point]
	label: Optional[str] = ''
	line: Optional[int] = Field(
		0,
		description='Stores the sequential line number of the para text starting from 1'
	)

	@classmethod
	def from_points(cls, points: List[Tuple[int, int]], label='', line=0):
		"""
		construct a Region class from the bounding box class
		"""
		return cls(
			points=[Point(*i) for i in points],
			label=label,
			line=line,
		)

	def to_polylines_pts(self):
		ret = [[i.x, i.y] for i in self.points]
		ret = np.array(ret)
		return ret.reshape(-1, 1, 2)


class Region(BaseModel):
	bounding_box: BoundingBox
	label: Optional[str] = ''
	line: Optional[int] = Field(
		0,
		description='Stores the sequential line number of the para text starting from 1'
	)

	def to_xyxy(self) -> Tuple[int, int, int, int]:
		return (
			self.bounding_box.x,
			self.bounding_box.y,
			self.bounding_box.x + self.bounding_box.w,
			self.bounding_box.y + self.bounding_box.h
		)

	def to_polygon(self) -> PolygonRegion:
		return PolygonRegion(
			points=self.bounding_box.to_polygon(),
			label=self.label,
			line=self.line
		)

	@classmethod
	def from_xyxy(cls, coords: Tuple[int, int, int, int], label='', line=0):
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


class LayoutResponse(BaseModel):
	regions: List[Region]


class LayoutImageResponse(BaseModel):
	"""
	Model class for holding the layout response for one single image
	"""
	image_name: str
	regions: List[Union[Region, PolygonRegion]]

	def to_polygon(self) -> 'LayoutImageResponse':
		ret = []
		for region in self.regions:
			if type(region) == Region:
				ret.append(region.to_polygon())
			else:
				ret.append(region)
		return LayoutImageResponse(
			image_name=self.image_name,
			regions=ret
		)
