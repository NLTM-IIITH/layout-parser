import json
import os

from helper import *
from models import *

print(os.listdir("data"))
#Perform inference and save output in output.json
script_inference_alexnet("data")