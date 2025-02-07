#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo $SCRIPT_DIR
# Define the relative paths based on the script's location
workdir=$(realpath "$SCRIPT_DIR/workdir")
echo $workdir
docker volume create \
 --opt type=none \
 --opt o=bind \
 --opt device=$workdir \
 MLFlow

docker container run \
 --name mlflow\
 --mount source=MLFlow,target=/MLFlow \
 --env-file "$SCRIPT_DIR/.env_git"\
 -p 5000:5000\
 --network movie_recommendation_network\
 --rm\
 mlflow:latest