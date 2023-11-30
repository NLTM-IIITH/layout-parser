#!/bin/bash

DATA_DIR="$1"
SI_VENV="$2"
VENV="$3"
echo "Checking for data dir"
if [ ! -d "$DATA_DIR" ]; then
	echo "$DATA_DIR : Enter a valid data directory"
	exit
else
	echo -e "DATA_DIR\t$DATA_DIR"
fi

if [ ! -d "$SI_VENV" ]; then
	echo "$SI_VENV : Enter a valid path to script-identification virtual environment"
	exit
else
	echo -e "Script-identification virtual environment path:\t$SI_VENV"
fi

if [ ! -d "$VENV" ]; then
	echo "$VENV : Enter a valid path to virtual environment"
	exit
else
	echo -e "Virtual environment path:\t$VENV"
fi

source "$SI_VENV/bin/activate"
echo "Activated script-identification virtual environment"
python ./server/modules/script_identification/infer.py $DATA_DIR
source "$VENV/bin/activate"
echo "Activated main virtual environment"
