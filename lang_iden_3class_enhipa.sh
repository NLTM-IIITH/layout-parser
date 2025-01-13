#!/bin/bash

DATA_DIR="$1"

MODEL_DIR="/home/layout/models/language_identification_3class_enhipa"

if [ ! -d "$DATA_DIR" ]; then
	echo "$DATA_DIR : Enter a valid data directory"
	exit
fi


if [ ! -d "$MODEL_DIR" ]; then
	echo "$MODEL_DIR : Enter a valid model directory"
	exit
fi

docker run --rm --gpus all --net host \
	-v $DATA_DIR:/data \
	-v $MODEL_DIR:/model \
	langiden:3class-enhipa \
	python infer.py
