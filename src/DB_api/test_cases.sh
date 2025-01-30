#!/bin/bash

# Base URL of the API
BASE_URL="http://localhost:8000"

# Test Home Endpoint
echo "Testing Home Endpoint"
curl -X GET "${BASE_URL}/"
echo -e "\n"

# Test Movies Endpoints
echo "Testing Movies Endpoints"
curl -X GET "${BASE_URL}/api/v1/movies"
echo -e "\n"

echo "Filtering movies by movieid"
curl -X GET "${BASE_URL}/api/v1/movies/filter?movieid=1"
echo -e "\n"

echo "Filtering movies by title"
curl -X GET "${BASE_URL}/api/v1/movies/filter?title=Toy+Story"
echo -e "\n"

echo "Filtering movies by genres"
curl -X GET "${BASE_URL}/api/v1/movies/filter?genres=Adventure"
echo -e "\n"

echo "Filtering movies by multiple parameters"
curl -X GET "${BASE_URL}/api/v1/movies/filter?movieid=1&title=Toy+Story"
echo -e "\n"

curl -X GET "${BASE_URL}/api/v1/movies/filter?title=Toy+Story&genres=Adventure"
echo -e "\n"

curl -X GET "${BASE_URL}/api/v1/movies/filter?movieid=1&genres=Adventure"
echo -e "\n"

curl -X GET "${BASE_URL}/api/v1/movies/filter?movieid=1&title=Toy+Story&genres=Adventure"
echo -e "\n"

# Test Ratings Endpoints
echo "Testing Ratings Endpoints"
curl -X GET "${BASE_URL}/api/v1/ratings"
echo -e "\n"

echo "Filtering ratings by userid"
curl -X GET "${BASE_URL}/api/v1/ratings/filter?userid=1"
echo -e "\n"

echo "Filtering ratings by movieid"
curl -X GET "${BASE_URL}/api/v1/ratings/filter?movieid=1"
echo -e "\n"

echo "Filtering ratings by rating"
curl -X GET "${BASE_URL}/api/v1/ratings/filter?rating=5.0"
echo -e "\n"

echo "Filtering ratings by multiple parameters"
curl -X GET "${BASE_URL}/api/v1/ratings/filter?userid=1&movieid=1"
echo -e "\n"

curl -X GET "${BASE_URL}/api/v1/ratings/filter?userid=1&rating=5.0"
echo -e "\n"

curl -X GET "${BASE_URL}/api/v1/ratings/filter?movieid=1&rating=5.0"
echo -e "\n"

curl -X GET "${BASE_URL}/api/v1/ratings/filter?userid=1&movieid=1&rating=5.0"
echo -e "\n"

# Test Tags Endpoint
echo "Testing Tags Endpoint"
curl -X GET "${BASE_URL}/api/v1/tags"
echo -e "\n"

# Test Database Creation Endpoint
echo "Testing Database Creation Endpoint (remove existing)"
curl -X POST "${BASE_URL}/api/v1/database/create?remove_existing=true"
echo -e "\n"

echo "Testing Database Creation Endpoint (keep existing)"
curl -X POST "${BASE_URL}/api/v1/database/create?remove_existing=false"
echo -e "\n"