from typing import List, Optional

from pydantic import BaseModel, Field


class LayoutConfig(BaseModel):
	pass

class ImageFile(BaseModel):
	imageContent: Optional[str] = Field(
		description='image content',
	)
	imageUri: Optional[str] = Field(
		description='path on gcp/s3 bucket or https url',
	)

class LayoutRequest(BaseModel):
	image: List[ImageFile]
	config: LayoutConfig


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

	def topolygon(self) -> List[List[int]]:
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
