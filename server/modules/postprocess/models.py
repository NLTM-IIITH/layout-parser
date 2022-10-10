from typing import List

from pydantic import BaseModel, Field

# SI stands for Script Identification

class SIRequest(BaseModel):
	images: List[str] = Field(
		...,
		description='List of images in base64 format'
	)


class SIResponse(BaseModel):
	text: str
