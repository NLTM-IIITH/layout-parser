### Path to the Data and Results directories
# NOTE : ALL THE PATHS ARE RELATIVE TO THE LAYOUT-PARSER-API
# this is the folder where all the input images are saved.
IMAGE_FOLDER = '/home/venkat/Projects/workbook/layout-parser-api/images'

#  Textron folders
INPUT_DATA_DIR = ''          # input dir where all the input files from the request are saved 
RESULTS_DATA_DIR  = 'server/modules/textron_results'   # output Dir where textron results are to be saved
PRED_IMAGES_FOLDER='server/modules/textron_results/predictions/devanagari'    # output predicted images from textron 
PRED_TXT_FOLDER='server/modules/textron_results/txt/devanagari'   # output txt files containing the bboxes from textron
PRED_CAGE_FOLDER='server/modules/textron_results/cage/devanagari'  # output cage files 
TEXTRON_MAIN_FILE='server.modules.textron.main'       # a path to the main.py of the textron to be run - in this format because other this were giving relative import error
TEXTRON_DEPENDENCIES='server/modules/textron_results/textron_dependencies/'