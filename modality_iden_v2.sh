#!/bin/bash

DATA_DIR="$1"

echo "Checking for data dir"
if [ ! -d "$DATA_DIR" ]; then
	echo "$DATA_DIR : Enter a valid data directory"
	exit
else
	echo -e "DATA_DIR\t$DATA_DIR"
fi

docker run --rm --gpus all --net host \
	-v $DATA_DIR:/data \
	modalityiden:ac-v1 \
	python infer.py
