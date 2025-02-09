#!/bin/bash
docker exec -u 0 -it mlflow_container /bin/bash -c "
conda run -p ./conda_envs/mlflow_env mlflow gc;
sudo rm -rf ./artifacts/*
exit
"