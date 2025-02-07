#!/bin/bash
# Make sure the script directory is exported as an environment variable
export SCRIPT_DIR=$(dirname "$(realpath "$0")")

Run Docker Compose
docker system prune -f -a
docker build prune -f
docker image prune -f
docker container prune -f
docker volume prune -f
docker-compose up --build -d #--no-cache

# Base URL of the API
# BASE_URL="http://localhost:8000"
# echo "Testing Database Creation Endpoint (remove existing)"
# curl -X POST "${BASE_URL}/api/v1/database/create?remove_existing=true"
# echo -e "\n"