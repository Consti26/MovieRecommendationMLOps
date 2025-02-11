#!/bin/bash
# Make sure the script directory is exported as an environment variable
docker stop $(docker ps -a -q)
export SCRIPT_DIR=$(dirname "$(realpath "$0")")

mkdir -p "${SCRIPT_DIR}/models/mlflow/workdir"
mkdir -p  "${SCRIPT_DIR}/data/raw_data"
mkdir -p  "${SCRIPT_DIR}/data/processed_data"

#############################################
# If you run for the first time, uncomment! #
#############################################

sudo rm -rf ${SCRIPT_DIR}/models/mlflow/workdir/*
sudo rm -rf ${SCRIPT_DIR}/models/mlflow/workdir/.*
sudo rm -rf ~/.local/share/Trash/*
sudo rm -rf ${SCRIPT_DIR}/data/raw_data/*
docker system prune -f -a
docker volume prune -f
docker-compose up --build -d #--no-cache
Base URL of the API
BASE_URL="http://localhost:8000"
echo "Testing Database Creation Endpoint (remove existing)"
curl -X POST "${BASE_URL}/api/v1/database/create?remove_existing=true"
echo -e "\n"

#######################
# Any subsequent Run! #
#######################

# docker builder prune -f
# docker-compose up #-d 
# docker builder prune -f

