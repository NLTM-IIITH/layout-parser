from enum import Enum
from optparse import Option
from os.path import join
from typing import List, Optional

from pydantic import BaseModel, Field


class TemplateChoice(str, Enum):
	template1 = 'template1'
	template2 = 'template2'


class LayoutIn(BaseModel):
	image: str = Field(
		...,
		description='Give the Downloadable URL of the filled form image'
	)
	template_image: str = Field(
		...,
		description='URL to the template image file (either JPG or PNG)'
	)
	template_coords: str = Field(
		...,
		description='URL to the CSV encoded Bounding boxes for template'
	)

class LayoutOut(BaseModel):
	images: List[str] = Field(
		[],
		description='List of extracted word images sorted by template bbox (in base64 format)'
	)
