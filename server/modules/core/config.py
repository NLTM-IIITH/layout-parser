### Path to the Data and Results directories
# NOTE : ALL THE PATHS ARE RELATIVE TO THE LAYOUT-PARSER-API
# this if the folder where all the input images are saved.
IMAGE_FOLDER = '/home/ishan/Desktop/Ishan/Documents/iitb/Bhashini App/Layout_New/layout-parser-api/server/modules/table/images'
=======
#  Textron folders
INPUT_DATA_DIR = '/home/venkat/Projects/workbook/layout-parser-api'          # input dir where all the input files from the request are saved 
RESULTS_DATA_DIR  = '/home/venkat/Projects/workbook/layout-parser-api/server/modules/textron/textron_results'   # output Dir where textron results are to be saved
PRED_IMAGES_FOLDER='/home/venkat/Projects/workbook/layout-parser-api/server/modules/textron/textron_results/predictions/devanagari'    # output predicted images from textron 
PRED_TXT_FOLDER='/home/venkat/Projects/workbook/layout-parser-api/server/modules/textron/textron_results/txt/devanagari'   # output txt files containing the bboxes from textron
PRED_CAGE_FOLDER='s/home/venkat/Projects/workbook/layout-parser-api/server/modules/textron/textron_results/cage/devanagari'  # output cage files 
TEXTRON_MAIN_FILE='server.modules.textron.main'       # a path to the main.py of the textron to be run - in this format because other this were giving relative import error
TEXTRON_DEPENDENCIES='/home/venkat/Projects/workbook/layout-parser-api/server/modules/textron/textron_results/textron_dependencies
