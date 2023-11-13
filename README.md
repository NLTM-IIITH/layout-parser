# Layout Parser

## Description

This is an API that facilitates the use Layout Parser models to detect and extract
the Bounding Box information from JPG images.

## Getting Started

### Dependencies

* Ubuntu
* Python 3.6+

### Starting FastAPI Server

* Clone the repo to local system
```
git clone https://github.com/NLTM-IIITH/layout-parser.git
```
* Create a python 3.6+ virtualenv
```
python -m venv layout-parser-venv
```
* Install the required python packages
```
pip install -r requirements.txt
```
* Simply run the main.py file
```
python main.py
```
* To access the swagger UI go to [http://127.0.0.1:8888](http://127.0.0.1:8888)

### Changes made for table detection module
#### helper.py
- Created helper function 'process_multiple_image_fasterrcnn()' and 'process_image_fasterrcnn()' which uses function 'get_tables_from_page()', imported from table_cellwise_detection.py, for inference.

#### models.py
- updated class 'ModelChoice' with the model 'fasterrcnn'.
- updated class 'Region' to support the response output of inference call for the 'fasterrcnn' model.

#### ocr_config.py
- created model as github release and added its url to 'faster_rcnn_model_url'.

#### routes.py
- created API endpoint '/table' which uses helper function 'process_multiple_image_fasterrcnn()'.
- created API endpoint '/visualize/table' which uses helper function 'process_image_fasterrcnn()' and uses cv2 to display bounding box of table and cells as well.

#### table_cellwise_detection.py
- added model script for inference of fasterrcnn.
- updated model weights loading method from 'faster_rcnn_model_url'.

#### app.py
- added app.include_router(table_router).

#### requirements.txt
- added following requirements:
 - pytesseract
 - pdf2image
 - layoutparser
 - bs4
 - torchg
 - torchvision

### Example
- layout/table/
  ![image](https://github.com/Biyani404198/layout-parser-api/assets/92304955/2c3f4a06-c77b-4856-8f25-ad75c2ece25f)
  ![image](https://github.com/Biyani404198/layout-parser-api/assets/92304955/bc32c3bf-3a13-4c5c-a60f-7a015d942ab3)
  ![image](https://github.com/Biyani404198/layout-parser-api/assets/92304955/297f08a8-36ba-446c-bd64-9901c91fe35c)

- layout/visualize/table/
  ![image](https://github.com/Biyani404198/layout-parser-api/assets/92304955/be8ecb12-437b-4cd6-9a22-c6d339afb282)
  ![image](https://github.com/Biyani404198/layout-parser-api/assets/92304955/0dbed7ea-bfd8-4a54-a041-4d3a017dc613)





## Authors

Krishna Tulsyan
[LinkedIn](https://www.linkedin.com/in/krishna-tulsyan/)

<!-- ## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details -->

## Acknowledgments

* [docTR](https://github.com/mindee/doctr)
* [README.md Template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
