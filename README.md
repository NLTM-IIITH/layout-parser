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

### Changes made for script-identification model integration:
* Additional code is defined in script_identification module (server>modules>script_identification)
* script-identification request endpoint updated in server>modules>postprocess>routes.py to accomodate ModelChoice as additional input and to accomodate model selection
* Endpoint function executes shell script script_iden_iitb.sh, which calls infer.py
* In script_identification module:
  * infer.py performs inference and saves output as output.json
  * helper.py has function process_output which reads output.json and returns output in response format
  * models.py defines the response format and model-choice class using pydantic BaseModel
* Dependencies are saved in requirements_script_identification.txt which adds onto requirements.txt with additional dependencies required
## Authors

Krishna Tulsyan
[LinkedIn](https://www.linkedin.com/in/krishna-tulsyan/)

<!-- ## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details -->

## Acknowledgments

* [docTR](https://github.com/mindee/doctr)
* [README.md Template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)