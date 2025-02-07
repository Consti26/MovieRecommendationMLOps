#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo "Script directory: $SCRIPT_DIR"

# Define directory paths
PROCESSED_DATA_PATH="$SCRIPT_DIR/processed_data"
RAW_DATA_PATH="$SCRIPT_DIR/raw_data"

# Ensure directories exist BEFORE using realpath
mkdir -p "$PROCESSED_DATA_PATH"
mkdir -p "$RAW_DATA_PATH"

# Define the relative paths based on the script's location
PROCESSED_DATA_PATH=$(realpath "$SCRIPT_DIR/processed_data")
RAW_DATA_PATH=$(realpath "$SCRIPT_DIR/raw_data")

echo "Processed Data Path: $PROCESSED_DATA_PATH"
echo "Raw Data Path: $RAW_DATA_PATH"

# Build Docker image
docker image build . -t api_database:latest