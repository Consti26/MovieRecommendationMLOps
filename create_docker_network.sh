#!/bin/bash

docker network create --subnet 172.50.0.0/16 --gateway 172.50.0.1 movie_recommendation_network
docker network ls