import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow.tracking import MlflowClient

# Retrieve environment variables with defaults
DATABASE_PORT = os.getenv("DATABASE_PORT", "8000")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5040")
PREPROCESSING_PORT = os.getenv("PREPROCESSING_PORT", "9000")
TRAINING_PORT = os.getenv("TRAINING_PORT", "8080")
INFERENCE_PORT = os.getenv("INFERENCE_PORT", "8090")

DATABASE_CONTAINER = os.getenv("DATABASE_CONTAINER", "database_container")
MLFLOW_CONTAINER = os.getenv("MLFLOW_CONTAINER", "mlflow_container")
# If needed, define container names for the other services
PREPROCESSING_CONTAINER = os.getenv("PREPROCESSING_CONTAINER", "preprocess_container")
TRAINING_CONTAINER = os.getenv("TRAINING_CONTAINER", "training_container")
INFERENCE_CONTAINER = os.getenv("INFERENCE_CONTAINER", "inference_container")

# Construct API URLs using the environment variables
PREPROCESSING_API_URL = f"http://{PREPROCESSING_CONTAINER}:{PREPROCESSING_PORT}/preprocess_data"
TRAINING_API_URL = f"http://{TRAINING_CONTAINER}:{TRAINING_PORT}/train_content_filter"
CHECK_TABLES_API_URL = f"http://{DATABASE_CONTAINER}:{DATABASE_PORT}/api/v1/database/check_tables"
CREATE_DATABASE_API_URL = f"http://{DATABASE_CONTAINER}:{DATABASE_PORT}/api/v1/database/create"
MLFLOW_URL = f"http://{MLFLOW_CONTAINER}:{MLFLOW_PORT}"
FETCH_NEW_MODEL_API_URL = f"http://{INFERENCE_CONTAINER}:{INFERENCE_PORT}/fetch_new_model"

# Définition des constantes API
#PREPROCESSING_API_URL = "http://preprocess_container:9000/preprocess_data"
#TRAINING_API_URL = "http://training_container:8080/train_content_filter"
#CHECK_TABLES_API_URL = "http://database_container:8000/api/v1/database/check_tables"
#CREATE_DATABASE_API_URL = "http://database_container:8000/api/v1/database/create"
#MLFLOW_URL = "http://mlflow_container:5040"
#FETCH_NEW_MODEL_API_URL = "http://inference_container:8090/fetch_new_model"

# Fonction pour appeler l'API de preprocessing
def call_preprocessing_api():
    print("starting preprocessing test")
    print(PREPROCESSING_API_URL)
    response = requests.post(PREPROCESSING_API_URL)
    print("api request is ok")
    print(response)
    if response.status_code == 200:
        print("✅ Preprocessing terminé avec succès !")
    else:
        raise Exception(f"❌ Erreur lors du preprocessing : {response.text}")

# Fonction pour appeler l'API d'entraînement
def call_training_api():
    params = {
    "labels": {
        "experiment_name": "Train_Contentbased_Filter",
        "model_name": "contentbased_filter"
    },
    "tfidf_args": {
        "stop_words": "english",
        "max_features": 10000,
        "max_df": 1,
        "min_df": 1,
        "ngram_range": [1, 2]
    },
    "sample_fraction": 0.1
    }
    response = requests.post(TRAINING_API_URL, json=params)
    if response.status_code == 200:
        print("✅ Entraînement terminé avec succès !")
    else:
        raise Exception(f"❌ Erreur lors de l'entraînement : {response.text}")

def check_tables():
    response = requests.get(CHECK_TABLES_API_URL)
    if response.status_code == 200:
        tables_status = response.json()
        if tables_status['movies'] == 'exists' and tables_status['preprocessed_dataset'] == 'exists':
            return 'skip_creation_and_preprocessing'
        elif tables_status['movies'] == 'exists':
            return 'preprocess_data'
        else:
            return 'create_database'
    else:
        raise Exception(f"❌ Erreur lors de la vérification des tables : {response.text}")

def create_database():
    response = requests.post(CREATE_DATABASE_API_URL)
    if response.status_code == 200:
        print("✅ Base de données créée avec succès !")
    else:
        raise Exception(f"❌ Erreur lors de la création de la base de données : {response.text}")

def tag_best_model():
    mlflow.set_tracking_uri(MLFLOW_URL)
    client = MlflowClient()

    experiment_name = "Train_Contentbased_Filter"
    model_name= "contentbased_filter"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.vocabulary_size DESC"]
    )

    if not runs:
        raise Exception("No successful runs found.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"

    model_details = mlflow.register_model(model_uri, model_name)
    model_version = model_details.version

    # Remove tags from all other models
    all_models = client.search_model_versions(f"name='{model_name}'")
    for model in all_models:
        if model.version != model_version:
            client.delete_model_version_tag(model.name, model.version, "stage")
            client.delete_model_version_tag(model.name, model.version, "performance")

    # Tag the best model
    client.set_model_version_tag(model_details.name, model_version, "stage", "production")
    client.set_model_version_tag(model_details.name, model_version, "performance", "champion")

    print(f"Model {model_details.name} version {model_version} tagged as 'stage:production' and 'performance:champion'")

def fetch_new_model():
    request_body = {
        # Include the necessary fields for the request body here
        "model_name": "contentbased_filter",
        "stage": "production"
    }
    response = requests.post(FETCH_NEW_MODEL_API_URL, json=request_body)
    if response.status_code == 200:
        print("✅ New model fetched successfully!")
    else:
        raise Exception(f"❌ Error fetching the new model: {response.text}")

# Définition du DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 12),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Pipeline MLOps orchestrée avec Airflow',
    schedule_interval=timedelta(days=1),  # Exécution quotidienne
    catchup=False,
    tags=['movie_recommendation']
)

# Définition des tâches
check_tables_task = BranchPythonOperator(
    task_id='check_tables',
    python_callable=check_tables,
    dag=dag,
)

create_database_task = PythonOperator(
    task_id='create_database',
    python_callable=create_database,
    dag=dag,
)

# Définition des tâches
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=call_preprocessing_api,
    dag=dag,
)

skip_creation_and_preprocessing = DummyOperator(
    task_id='skip_creation_and_preprocessing',
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=call_training_api,
    dag=dag,
    trigger_rule='one_success'
)

tag_best_model_task = PythonOperator(
    task_id='tag_best_model',
    python_callable=tag_best_model,
    dag=dag,
)

fetch_new_model_task = PythonOperator(
    task_id='fetch_new_model',
    python_callable=fetch_new_model,
    dag=dag,
)

# Orchestration du DAG
check_tables_task >> [create_database_task, preprocess_task, skip_creation_and_preprocessing]
create_database_task >> preprocess_task
preprocess_task >> train_task
skip_creation_and_preprocessing >> train_task
train_task >> tag_best_model_task >> fetch_new_model_task
