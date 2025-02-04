#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo $SCRIPT_DIR
# Define the relative paths based on the script's location
PROCESSED_DATA_PATH=$(realpath "$SCRIPT_DIR/processed_data")
RAW_DATA_PATH=$(realpath "$SCRIPT_DIR/raw_data")

docker container run \
 --name db_api\
 --volume "$PROCESSED_DATA_PATH:/home/jovyan/DB_api/processed_data" \
 --volume "$RAW_DATA_PATH:/home/jovyan/DB_api/raw_data" \
 --env-file "$SCRIPT_DIR/.env_git"\
 -p 8000:8000\
 --rm\
 --hostname 'db_api'\
 db_api:latest