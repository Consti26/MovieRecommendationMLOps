#!/bin/bash
# Make sure the script directory is exported as an environment variable
docker stop $(docker ps -a -q)
export SCRIPT_DIR=$(dirname "$(realpath "$0")")
sudo rm -rf ./models/mlflow/workdir/*
sudo rm -rf ./models/mlflow/workdir/.*
# sudo rm -rf ~/.local/share/Trash/*
sudo rm -rf ./data/raw_data/*

# Run Docker Compose
docker system prune -f -a
docker builder prune -f
docker image prune -f
docker container prune -f
# docker volume rm mlflow_volume
docker volume prune -f
# docker-compose up --build -d #--no-cache


docker-compose up -d 
docker system prune -f -a
# sudo rm -rf /tmp/*.tmp

# Base URL of the API
BASE_URL="http://localhost:8000"
echo "Testing Database Creation Endpoint (remove existing)"
curl -X POST "${BASE_URL}/api/v1/database/create?remove_existing=true"
echo -e "\n"