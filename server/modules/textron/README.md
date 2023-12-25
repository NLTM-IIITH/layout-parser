# Textron

## Module Added
```

├── server/modules/textron
|   ├── helper.py                       # consists of inference code and the code for visualizaiton
|   ├── models.py                       # consists of required input/output structure


```
Added the routes in `server/modules/main/routes.py` in `doctr_layout_parser` and `layout_parser_swagger_only_demo`.
## Load from Docker Hub
```
docker pull shouryatyagi222/textron:1
```
## Docker command used :
```
docker run --rm --net host -v IMAGE_FOLDER:/data textron:1
```
Where `IMAGE_FOLDER` is the path in `server/modules/core/config.py`.
The `IMAGE_FOLDER` is the path where the Input Images are saved and the json output are saved.
### Textron Inference : /layout/
### Textron Visualization : /layout/visualize/

## Note
The Following Module downloads the required doctr models in the build image.
