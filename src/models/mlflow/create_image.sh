#!/bin/bash
# Get the directory where the script is located

docker builder prune -f
docker image prune -f
docker image build -t mlflow:latest .