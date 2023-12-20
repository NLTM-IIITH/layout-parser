# Textron

## Module Added
```

├── server/modules/textron
|   ├── helper.py                       # consists of inference code and the code for visualizaiton
|   ├── models.py                       # consists of required input/output structure


```
Added the routes in `server/modules/main/routes.py` in `doctr_layout_parser` and `layout_parser_swagger_only_demo`.
## Docker command used :
```
docker run --rm --net host -v IMAGE_FOLDER:/textron/data textron:1
```
Where IMAGE_FOLDER is the path in `server/modules/core/config.py`.
### Textron Inference : /layout/
### Textron Visualization : /layout/visualize/
