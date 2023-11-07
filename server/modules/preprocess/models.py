from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

class ModelChoice(str, Enum):
	doctr = 'doctr'
	tesseract = 'tesseract'

class ModalityChoice(str, Enum):
	word = 'word'
	line = 'line'
	page = 'page'

class LayoutConfig(BaseModel):
	pass

class ImageFile(BaseModel):
	imageContent: Optional[str] = Field(
		description='image content',
	)
	imageUri: Optional[str] = Field(
		description='path on gcp/s3 bucket or https url',
	)
class Image(BaseModel):
	""" List of """
	image: Optional[str] = Field(
		description='image name',
	)
class ImageColors(BaseModel):
	""" List of """
	image: Optional[str] = Field(
		description='image name',
	)
	rgbaValues: Optional[str] = Field(
		description='rgba values of the given image',
	)
	hexacode: Optional[str] = Field(
		description='hexa-code of the given image',
	)
class ImageFonts(BaseModel):
	""" List of """
	image: Optional[str] = Field(
		description='image name',
	)
	fontSize: Optional[str] = Field(
		description='Font-size of the text in the image',
	)
	fontFamily: Optional[str] = Field(
		description='Font-family of the given image',
	)
	fontDecoration: Optional[str] = Field(
		description='Font-Decoration of the given image',
	)
class ImageProperties(BaseModel):
	""" List of """
	image: Optional[str] = Field(
		description='image name',
	)
	tables: Optional[bool] = Field(
		description='Returns whether the document contains Tables',
	)
	columns: Optional[str] = Field(
		description='Returns column type of the page',
	)
	




class LayoutRequest(BaseModel):
	image: List[ImageFile]
	config: LayoutConfig
class preProcessorRequest(BaseModel):
	image: List[ImageFile]


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
class PreProcessorBinarizeResponse(BaseModel):
	images: List[Image]
class PreProcessorGrayScaleResponse(BaseModel):
	images: List[Image]
class PreProcessorColorResponse(BaseModel):
	images: List[ImageColors]
class PreProcessorFontResponse(BaseModel):
	images: List[ImageFonts]
class PreProcessorPropertiesResponse(BaseModel):
	images: List[ImageProperties]

class FontRegion(BaseModel):
	bounding_box: BoundingBox
	fontSize: Optional[int] = Field(description='size of the font')
	fontColor: Optional[List[int]] = Field(description='font color in RGB')
	fontDecoration: Optional[str] = Field(description='font decoration bold/regular')

class FontAttributeImage(BaseModel):
	image: Optional[str] = Field(description='image name')
	font_regions: List[FontRegion]

class FontAttributesResponse(BaseModel):
	images: List[FontAttributeImage]

