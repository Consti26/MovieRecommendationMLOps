#!/bin/bash

docker stop $(docker ps -a -q)
# docker rmi $(docker images -q) -f

# docker image rm mlfow:latest
# sudo rm -rf /home/ubuntu/oct24_cmlops_int_movie_reco/src/models/mlflow/workdir/*
# sudo rm -rf /home/ubuntu/oct24_cmlops_int_movie_reco/src/data/raw_data/*


docker network prune -f
docker volume prune -f

# docker image rm api_train_content:latest
docker system prune -f
docker volume rm mlflow_volume

if docker images --format '{{.Repository}}:{{.Tag}}' | grep -w "api_database:latest" > /dev/null
then
  echo "It exists. Skip the Database build..."
else
  echo "Building new Database image.."
  # Build Docker image
  docker build -t api_database:latest ./data -f /home/ubuntu/oct24_cmlops_int_movie_reco/src/data/dockerfile
fi

docker network create test_network

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo "Script directory: $SCRIPT_DIR"
# Define the relative paths based on the script's location
PROCESSED_DATA_PATH=$(realpath "$SCRIPT_DIR/data/processed_data")
RAW_DATA_PATH=$(realpath "$SCRIPT_DIR/data/raw_data")

echo "Processed data directory: $PROCESSED_DATA_PATH"
echo "Raw data directory: $RAW_DATA_PATH"

docker container run -d \
 --name database_container\
 --volume "$PROCESSED_DATA_PATH:/home/api_database/processed_data" \
 --volume "$RAW_DATA_PATH:/home/api_database/raw_data" \
 --env-file "$SCRIPT_DIR/data/.env_git"\
 --network test_network \
 -p 8000:8000\
 --rm\
 api_database:latest


##########################################
#             PREPROCESSING              #
##########################################

if  docker images --format '{{.Repository}}:{{.Tag}}' | grep -w "api_preprocess_content:latest" > /dev/null
then
  echo "It exists. Skip the Preprocess build..."
else
  echo "Building new Preprocess image.."
  # Build Docker image
  docker build -t api_preprocess_content:latest ./features
fi

docker container run -d \
 --name preprocess_container\
 --env-file "$SCRIPT_DIR/features/.env_git"\
 --network test_network \
 -p 9000:9000\
 --rm\
 api_preprocess_content:latest
##########################################
#             Create Volume              #
##########################################

docker volume create \
 --name mlflow_volume \
 --opt type=none \
 --opt o=bind \
 --opt device=/home/ubuntu/oct24_cmlops_int_movie_reco/src/models/mlflow/workdir


##########################################
#                 MLFOW                  #
##########################################

if  docker images --format '{{.Repository}}:{{.Tag}}' | grep -w "mlfow:latest" > /dev/null
then
  echo "It exists. Skip the MLFlow build..."
else
  echo "Building new MLFlow image.."
  # Build Docker image
  docker build -t mlfow:latest ./models/mlflow
fi

docker container run -d \
 --name mlflow_container\
 --env-file "$SCRIPT_DIR/models/mlflow/.env_git"\
 --mount source=mlflow_volume,target=/mlflow \
 --network test_network \
 -p 5000:5000\
 --rm\
 mlfow:latest

##########################################
#             Train                      #
##########################################

if  docker images --format '{{.Repository}}:{{.Tag}}' | grep -w "api_train_content:latest" > /dev/null
then
  echo "It exists. Skip the Training build..."
else
  echo "Building new Training image.."
  # Build Docker image
  docker build -t api_train_content:latest ./models/training
fi

docker container run \
 --name training_container\
 --env-file "$SCRIPT_DIR/models/training/.env_git"\
 --mount source=mlflow_volume,target=/mlflow \
 --network test_network \
 -p 8080:8080\
 --rm\
 api_train_content:latest


##########################################
#         Some Test Commands             #
##########################################

DBFILE="$PROCESSED_DATA_PATH/movielens.db"

if [ -f $DBFILE ]; then
   echo "DB $DBFILE exists.\n Skipping DB creation."
else
   echo "DB $DBFILE does not exist."
   curl -X POST http://localhost:8000/api/v1/database/create?remove_existing=true
fi