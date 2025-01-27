#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo $SCRIPT_DIR
# Define the relative paths based on the script's location
PROCESSED_DATA_PATH=$(realpath "$SCRIPT_DIR/processed_data")
RAW_DATA_PATH=$(realpath "$SCRIPT_DIR/raw_data")

mkdir $PROCESSED_DATA_PATH
mkdir $RAW_DATA_PATH

docker image build . -t db_api:latest