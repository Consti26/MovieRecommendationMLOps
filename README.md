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
    ├── create_docker_network.sh     # Script to create a Docker network
    ├── launch_streamlit.sh          # Script to launch the Streamlit UI    
    └── streamlit_app.py         # Streamlit interactive application

--------

Launching the projet 
------------

Follow the steps below to get the project up and running:

1. **Start the API Server**  
   Navigate to the `src` folder and run:
   ```bash
   bash launch_api.sh

2.	**Launch Airflow**
    Navigate to the airflow folder and run:
    ```bash
    docker-compose up --build

3. **Trigger DAG on airflow UI**
    Once the services are running, open your browser and go to: http://localhost:8081/home
   
4. **Start the Streamlit Application**
    Navigate back to the project root folder and run:
    ```bash
    bash launch_streamlit.sh

5. **Query the application**
    Once the Streamlit UI is loaded, go to the Demonstration page to query the application.

------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
