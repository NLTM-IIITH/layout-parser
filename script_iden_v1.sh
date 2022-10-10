#!/bin/bash

DATA_DIR="/home/layout/layout-parser/images"

echo "Checking for data dir"
if [ ! -d "$DATA_DIR" ]; then
	echo "$DATA_DIR : Enter a valid data directory"
	exit
else
	echo -e "DATA_DIR\t$DATA_DIR"
fi

docker run --rm --name=script-indentification --gpus all --net host \
	-v $DATA_DIR:/data \
	script:test \
	python app.py
