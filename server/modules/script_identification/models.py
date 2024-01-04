from pydantic import BaseModel, Field
from enum import Enum
class ModelChoice(str, Enum):
    alexnet = 'iitb-script-identification'
    default = 'default'	#Temporarily calling it default due to lack of knowledge regarding nature of model
    
class SIResponse(BaseModel):
	text: str = Field(
		...,
		description=(
			'This field contains the identified language/script for the image. '
			'this can take one of the 11 values. '
			"devanagari",
   			"bengali",
      		"gujarati",
        	"gurumukhi",
         	"kannada",
          	"malayalam",
           	"odia",
            "tamil",
            "urdu",
            "latin",
            "odia"
		)
	)

