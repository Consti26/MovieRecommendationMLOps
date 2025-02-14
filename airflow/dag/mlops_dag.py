from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import time

# Définition des constantes API
PREPROCESSING_API_URL = "http://preprocess_container:9000/preprocess_data"
TRAINING_API_URL = "http://training_container:8080/train_content_filter"

# Fonction pour appeler l'API de preprocessing
def call_preprocessing_api():
    response = requests.post(PREPROCESSING_API_URL)
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

# Définition du DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 12),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Pipeline MLOps orchestrée avec Airflow',
    schedule_interval=timedelta(days=1),  # Exécution quotidienne
    catchup=False,
)

# Définition des tâches
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=call_preprocessing_api,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=call_training_api,
    dag=dag,
)

# Orchestration du DAG
preprocess_task >> train_task
