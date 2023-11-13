# Textron : Layout API

## Routes added app.py:
`textron_router` - from modules/textron_api which consists of the inference code to get the bboxes from the textron .
`doctr_router` - from modules/doctr_api which consists of the doctr inference code to get the bboxes (no change from the code in inference code of modules/main)

## textron_api :
1. dependencies.py - from modules/main
2. helper.py - process_images_textron function for inference and textron_visulaize for visualization.
3. model.py -from modules/main containes Response and Request Structure for layout parser.
4. routes.py - containes call for textron reference (/layout/textron) and textron visualization (/layout/textron_visualizaiton)
TEXTRON FILES:
1. main.py - main file for running the textron and giving output
2. server/ , src/ , utilities/ , spear/ are the textron dependencies.

Possible Issues:
1. Textron requires the paths to a input folder having a images folder and a output folder path which is server/modules/textron_results it is a necessary folder as it has a params file which is required for inference.
2. all required paths are in server/modules/core/config.py - all the paths in the config are relative to the layout-parser-api folder and are necessary to be looked out for.

## folders added 
1. /images/ - all the input files will come in the folder.
2. /server/modules/textron_results - all the output from the textron will come in this folder.

## doctr_api : 
1. dependencies.py -from modules/main
2. helper.py - process_images_textron function for inference (same as in modules/main)
3. model.py -from modules/main
4. routes.py - uses '/' calls doctr_layout_parser '/visualization' is the visualization code (modules/main reference)