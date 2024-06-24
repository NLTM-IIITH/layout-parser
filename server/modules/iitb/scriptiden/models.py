from enum import Enum

from pydantic import BaseModel, Field


class PostprocessRequest(BaseModel):
	images: list[str] = Field(
		...,
		description='List of images in base64 format'
	)

class ModelChoice(str, Enum):
    alexnet = 'iitb-script-identification'
    default = 'default'	#Temporarily calling it default due to lack of knowledge regarding nature of model
    
class SIResponse(BaseModel):
    text: str = Field(
        ...,
        description=(
            'This field contains the identified language/script for the image. '
            'this can take one of the 11 values. '
            "devanagari"
               "bengali"
              "gujarati"
            "gurumukhi"
             "kannada"
              "malayalam"
               "odia"
            "tamil"
            "urdu"
            "latin"
            "odia"
        )
    )
