The Great Movie Filter - Filtering the Cosmos of Cinema to Find Your Perfect Pick
==============================

In today’s data-driven landscape, personalized experiences are key to engaging users and driving business value. This project focuses on developing a recommender system embedded within an MLOps framework to ensure the solution is robust, scalable, and maintainable.


Project Organization
------------
├── LICENSE                     
├── README.md                    # Project documentation and overview
├── airflow                      # Airflow orchestration for the MLOps pipeline
│   ├── dags                   # DAG definitions
│   ├── docker-compose.yaml    # Docker Compose for Airflow
│   ├── dockerfile             # Dockerfile for Airflow container
│   ├── logs                   
│   ├── plugins               
│   └── requirements.txt       # Airflow-specific Python dependencies
├── create_docker_network.sh     # Script to create a Docker network
├── launch_streamlit.sh          # Script to launch the Streamlit UI
├── references                 # Documentation, diagrams, and reference materials
│   ├── airflow.png
│   ├── content_based.png
│   ├── database.png
│   ├── final_architecture.jpg
│   ├── inference.png
│   ├── initial_architecture.jpg
│   ├── movie_recommender.png
│   ├── preprocessing.png
│   ├── siren flowchart code.txt
│   ├── training.png
│   └── trends.png
├── requirements.txt           # Global project Python dependencies
└── src                        # Main source code
    ├── config               # Configuration files and settings
    ├── data                 # Data ingestion and database API
    │   ├── api_database.py
    │   ├── create_image.sh
    │   ├── dockerfile
    │   ├── requirements.txt
    │   └── run_container.sh
    ├── docker-compose.yml   # Docker Compose for src services
    ├── features             # Preprocessing
    │   ├── WIP
    │   │   ├── preprocessing_collaborative_filtering_mlflow.py
    │   │   └── preprocessing_content_based_mlflow.py
    │   ├── api_preprocess_content.py
    │   ├── dockerfile
    │   └── requirements.txt
    ├── launch_api.sh        # Script to launch the main API
    ├── models               # Model-related code (training, inference, MLflow)
    │   ├── WIP
    │   │   ├── modelling_collaborative_filtering_mlflow.py
    │   │   └── modelling_content_based_mlflow.py
    │   ├── inference        # Inference API and scripts
    │   │   ├── api_inference_content.py
    │   │   ├── create_image.sh
    │   │   ├── dockerfile
    │   │   └── requirements.txt
    │   ├── mlflow           # MLflow integration and tracking
    │   │   ├── create_image.sh
    │   │   ├── dockerfile
    │   │   ├── environment.yml
    │   │   ├── gc_mlflow.sh
    │   │   ├── run_container.sh
    │   │   └── workdir
    │   │       ├── artifacts
    │   │       ├── environment.yml
    │   │       ├── mlflow_sees_this_dir
    │   │       └── mlruns
    │   ├── tfidf_vectorizer_model  # Custom TF-IDF model module
    │   │   └── tfidf_vectorizer_model.py
    │   └── training         # Training API and scripts
    │       ├── api_train_content.py
    │       ├── create_image.sh
    │       ├── dockerfile
    │       └── requirements.txt
    ├── test_cases.sh        # Shell script for test cases
    ├── test_db_preprocess.sh # Script to test database preprocessing
    └── tests                # Additional test scripts
        ├── get_data.py
        ├── get_data_2.py
        └── write_data.py
└── streamlit_app.py         # Streamlit interactive application

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
