#!/bin/bash

DATA_DIR="$1"

MODEL_DIR="/home/layout/models/language_identification_3class_enhipa"

echo "Checking for data dir"
if [ ! -d "$DATA_DIR" ]; then
	echo "$DATA_DIR : Enter a valid data directory"
	exit
else
	echo -e "DATA_DIR\t$DATA_DIR"
fi


echo "Checking for model dir"
if [ ! -d "$MODEL_DIR" ]; then
	echo "$MODEL_DIR : Enter a valid model directory"
	exit
else
	echo -e "MODEL_DIR\t$MODEL_DIR"
fi

docker run --rm --gpus all --net host \
	-v $DATA_DIR:/data \
	-v $MODEL_DIR:/model \
	langiden:3class-enhipa \
	python infer.py
