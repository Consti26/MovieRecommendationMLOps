#!/bin/bash
# Make sure the script directory is exported as an environment variable

export SCRIPT_DIR=$(dirname "$(realpath "$0")")

# # Load environment variables from .env file
# if [ -f "${SCRIPT_DIR}/.env" ]; then
#     export $(cat "${SCRIPT_DIR}/.env" | grep -v '#' | awk '/=/ {print $1}')
# fi

# mkdir -p "${SCRIPT_DIR}/models/mlflow/workdir"
# mkdir -p  "${SCRIPT_DIR}/data/raw_data"
# mkdir -p  "${SCRIPT_DIR}/data/processed_data"

# #############################################
# # If you run for the first time, uncomment! #
# #############################################

# sudo rm -rf ${SCRIPT_DIR}/models/mlflow/workdir/*
# sudo rm -rf ${SCRIPT_DIR}/models/mlflow/workdir/.[^.]*
# sudo rm -rf ~/.local/share/Trash/*
# sudo rm -rf ${SCRIPT_DIR}/data/raw_data/*
# sudo rm -rf ${SCRIPT_DIR}/data/processed_data/*

# echo ${SCRIPT_DIR}

# # ###########################
# # #   clean-up Docker State #
# # ###########################

# # # Stop and remove all containers
# docker stop $(docker ps -a -q)
# docker rm $(docker ps -a -q)

# # Remove all images
# docker rmi -f $(docker images -a -q)

# # Remove all volumes
# docker volume rm $(docker volume ls -q)

# # Prune the system to remove unused data
# docker system prune -f

# # Restart Docker service
# if [ "$(uname)" == "Darwin" ]; then
#     # macOS
#     osascript -e 'quit app "Docker"'
#     open /Applications/Docker.app
#     while ! docker system info > /dev/null 2>&1; do sleep 1; done
# else
#     # Linux
#     sudo systemctl restart docker
# fi
# echo  "Docker cleanup and restart completed."

# ##################
# #  up Containers #
# ##################

# docker-compose up --build #--no-cache

# Base URL of the API (using the environment variable)
# BASE_URL="http://localhost:${DATABASE_PORT}"
# echo "Testing Database Creation Endpoint (remove existing)"
# curl -X POST "${BASE_URL}/api/v1/database/create?remove_existing=true"
# echo -e "\n"

#######################
# Any subsequent Run! #
#######################

# docker builder prune -f
docker network create movie_recommendation_network
docker-compose up --build #-d 
# docker builder prune -f
