#!/bin/bash

# Base URL of the API
BASE_URL="http://172.18.0.2:8000"
BASE_URL="localhost:8000"

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

echo "Creates (or recreates) the 'preprocessed_dataset' table with specified columns."
curl -X POST "${BASE_URL}/api/v1/database/create_preprocessed_table"
echo -e "\n"

# Test database preprocessed_dataset
echo "Create pre processed dataset table"
curl -X POST http://localhost:8000/api/v1/database/create_preprocessed_table
echo -e "\n"

echo "Feed pre processed dataset table"
curl -X POST http://localhost:8000/api/v1/database/insert_preprocessed_data \
     -H "Content-Type: application/json" \
     -d '[
            {
                "movieId": 1,
                "title_year": "Toy Story (1995)",
                "genres": "Adventure Animation Children Comedy Fantasy",
                "title": "Toy Story",
                "year": 1995
            },
            {
                "movieId": 2,
                "title_year": "Jumanji (1995)",
                "genres": "Adventure Children Fantasy",
                "title": "Jumanji",
                "year": 1995
            }
        ]'
echo -e "\n"

echo "Feed pre processed dataset table"
curl -X GET "http://localhost:8000/api/v1/preprocessed_dataset"
echo -e "\n"
