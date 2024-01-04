# Layout Parser- Script Identification API

## Description

This module facilitates the integration of IITB script-identification models using Docker.

### Setup instructions
* Shift directory
``` 
cd server/modules/script_identification
```
* Build docker image
```
docker build -t script-identification .
```

### Changes made for script-identification model integration:
* Additional code is defined in script_identification module (server>modules>script_identification)
* script-identification request endpoint updated in server>modules>postprocess>routes.py to accomodate ModelChoice as additional input and to accomodate model selection
* Endpoint function runs docker container
* In script_identification module:
  * infer.py performs inference and saves output as output.json
  * helper.py has function process_output which reads output.json and returns output in response format
  * models.py defines the response format and model-choice class using pydantic BaseModel
* Dependencies are saved in requirements_script_identification.txt which adds onto requirements.txt with additional dependencies required
* The model architecture is defined in server>modules>script_identification>iitb_script_identification_model.py
  
### To perform inference using API:
* CURL: curl -X 'POST' \
  'http://0.0.0.0:8888/layout/postprocess/script?model=iitb-script-identification' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "images": ["base 64 format image string"]
}'
## Authors

Krishna Tulsyan
[LinkedIn](https://www.linkedin.com/in/krishna-tulsyan/)

<!-- ## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details -->

## Acknowledgments

* [docTR](https://github.com/mindee/doctr)
* [README.md Template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)