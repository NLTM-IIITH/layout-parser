from pydantic import BaseModel, Field
# SI stands for Script Identification

class PostprocessRequest(BaseModel):
	images: list[str] = Field(
		...,
		description='List of images in base64 format'
	)

class ClassifyResponse(BaseModel):
	text: str


class SIResponse(BaseModel):
	text: str = Field(
		...,
		description=(
			'This field contains the identified language/script for the image. '
			'this can take one of the 14 values. '
			'"assamese", '
			'"bengali", '
			'"english", '
			'"gujarati", '
			'"punjabi", '
			'"hindi", '
			'"kannada", '
			'"malayalam", '
			'"manipuri", '
			'"marathi", '
			'"oriya", '
			'"tamil", '
			'"telugu", '
			'"urdu", '
		)
	)


# MI stands for Modality Identification

class MIResponse(BaseModel):
	text: str = Field(
		...,
		description=(
			'This field contains the identified modality for the image. '
			'this can take one of the 3 values. '
			'"printed", '
			'"handwritten", '
			'"scenetext", '
		)
	)