MLOps for Movie Recommender System 

==============================

This project focuses on developing a recommender system embedded within an MLOps framework to ensure the solution is robust, scalable, and maintainable performance of the operations.

Project Organization
------------

    ├── LICENSE                     
    ├── README.md                   <- The top-level README for developers using this project.
    ├── __pycache__                 
    ├── airflow                     <- Configuration & scripts for Apache Airflow
    │   ├── dags                    <- Airflow DAG definitions for ML pipelines
    │   │   ├── __pycache__         
    │   │   │   ├── __pycache__     
    │   │   │   ├── mlops_dag.py     
    │   │   │   └── mlops_dag.py    
    │   │   └── mlops_dag.py         <- Main DAG file defining the MLops pipeline
    │   ├── docker-compose-old.yml   
    │   ├── docker-compose.yaml      <- Current Docker Compose configuration for Airflow services
    │   ├── dockerfile               <- Dockerfile to build the Airflow container image
    │   ├── logs                     
    │   ├── plugins                  
    │   └── requirements.txt         <- Python dependencies required for Airflow
    ├── create_docker_network.sh      <- Shell script to create a Docker network for container communication
    ├── models                      <- Directory to store trained machine learning models
    ├── notebooks                   <- Jupyter notebooks for exploration and experimentation
    ├── references                  <- Documentation and images detailing architecture and design
    │   ├── Modelling_documentation.docx
    │   ├── content_based.png
    │   ├── database.png
    │   ├── final_architecture.jpg
    │   ├── inference.png
    │   ├── initial_architecture.jpg
    │   ├── movie_recommender.png
    │   ├── preprocessing.png
    │   ├── training.png
    │   └── trends.png
    ├── reports                     <- Generated reports and figures for analysis
    │   └── figures                <- Supporting images and charts
    ├── requirements.txt            <- Global Python dependencies for the entire project
    ├── src                         <- Main source code for the ML application
    │   ├── __init__.py             <- Initializes the src package
    │   ├── config                  <- Configuration files and settings for the application
    │   ├── data                    <- Data ingestion and processing module
    │   │   ├── api_database.py      <- Script for API interactions with the database
    │   │   ├── create_image.sh      <- Script to build a Docker image for the data module
    │   │   ├── dockerfile           <- Dockerfile for the data module container
    │   │   ├── processed_data       <- Folder for storing processed datasets (e.g., movielens.db)
    │   │   ├── raw_data             <- Folder for raw input datasets
    │   │   ├── requirements.txt     <- Dependencies specific to the data module
    │   │   └── run_container.sh     <- Script to run the data module container
    │   ├── docker-compose.yml       <- Docker Compose file for local development
    │   ├── features                <- Feature engineering code and utilities
    │   │   ├── WIP                  <- Work-In-Progress feature scripts
    │   │   ├── __pycache__          <- Cached Python files for features
    │   │   ├── api_preprocess_content.py  <- API endpoint for content-based preprocessing
    │   │   ├── dockerfile           <- Dockerfile for the features service container
    │   │   └── requirements.txt     <- Dependencies for feature engineering
    │   ├── launch_api.sh            <- Script to launch the ML API service
    │   ├── models                  <- ML model-related scripts and Docker configurations
    │   │   ├── WIP                  <- Work-In-Progress model scripts
    │   │   ├── inference            <- Inference module for serving models
    │   │   │   ├── api_inference_content.py  <- API endpoint for model inference
    │   │   │   ├── create_image.sh           <- Build Docker image for the inference service
    │   │   │   ├── dockerfile                <- Dockerfile for the inference container
    │   │   │   └── requirements.txt          <- Inference service dependencies
    │   │   ├── mlflow               <- MLflow integration for experiment tracking
    │   │   │   ├── create_image.sh   <- Build MLflow Docker image
    │   │   │   ├── dockerfile        <- Dockerfile for the MLflow container
    │   │   │   ├── environment.yml   <- Conda environment configuration for MLflow
    │   │   │   ├── gc_mlflow.sh      <- Script to manage/cleanup MLflow artifacts
    │   │   │   ├── run_container.sh  <- Run MLflow container
    │   │   │   └── workdir           <- Workspace for MLflow artifacts and run logs
    │   │   │       ├── artifacts     <- Artifacts generated during MLflow runs
    │   │   │       ├── environment.yml
    │   │   │       ├── mlflow_sees_this_dir  <- Placeholder detected by MLflow
    │   │   │       └── mlruns        <- MLflow run metadata and stored models
    │   │   ├── tfidf_vectorizer_model  <- TF-IDF vectorizer model code
    │   │   │   ├── __init__.py     <- Module initializer for the TF-IDF model
    │   │   │   └── tfidf_vectorizer_model.py  <- Code for building the TF-IDF vectorizer
    │   │   └── training            <- Training scripts for machine learning models
    │   │       ├── api_train_content.py  <- API endpoint to trigger model training
    │   │       ├── create_image.sh       <- Build Docker image for the training service
    │   │       ├── dockerfile            <- Dockerfile for the training container
    │   │       └── requirements.txt      <- Training module dependencies
    │   ├── test_cases.sh            <- Script to execute overall test cases
    │   ├── test_db_preprocess.sh    <- Script to test database preprocessing routines
    │   ├── tests                    <- Unit and integration tests for the project
    │   │   ├── get_data.py         <- Tests for data retrieval functions
    │   │   ├── get_data_2.py       <- Additional tests for data retrieval
    │   │   └── write_data.py       <- Tests for data writing functions
    │   └── visualization           <- Data visualization utilities and scripts
    │       ├── __init__.py         <- Initializes the visualization module
    │       └── visualize.py        <- Script to generate visual representations of data
    └── streamlit_app.py            <- Streamlit app script for an interactive UI
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. <-cookiecutterdatascience</small></p>
