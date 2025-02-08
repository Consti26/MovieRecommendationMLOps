import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
import socket
import os
import json
from pydantic import BaseModel
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

# Get the paths from environment variables
DATABASE_CONTAINER = os.getenv('DATABASE_CONTAINER', 'database_container')
DATABASE_PORT = int(os.getenv('DATABASE_PORT', '8000'))
MLFLOW_CONTAINER = os.getenv('MLFLOW_CONTAINER', 'mlflow_container')
MLFLOW_PORT = int(os.getenv('MLFLOW_PORT', '5000'))

try:
    # DNS-Lookup durchführen
    DATABASE_ADDRESS = socket.gethostbyname(DATABASE_CONTAINER)
    API_DATABASE_URL = 'http://{address}:{port}'.format(address=DATABASE_ADDRESS, port=DATABASE_PORT )
    print(f"Die URI von {DATABASE_CONTAINER} ist: {API_DATABASE_URL}")
except socket.gaierror as e:
    API_DATABASE_URL = 'http://localhost:8000'
    print(f"Fehler beim Abrufen der IP-Adresse für {API_DATABASE_URL}: {e}")

try:
    # DNS-Lookup durchführen MLFlow
    mlflow_address = socket.gethostbyname(MLFLOW_CONTAINER)
    MLFLOW_URL = 'http://{address}:{port}'.format(address=mlflow_address, port=MLFLOW_PORT )
    print(f"Die URI von {MLFLOW_CONTAINER} ist: {MLFLOW_URL}")
except socket.gaierror as e:
    MLFLOW_URL = 'http://localhost:5000'
    print(f"Fehler beim Abrufen der IP-Adresse für {MLFLOW_CONTAINER}: {e}")


app = FastAPI()

class TrainingParams(BaseModel):
    max_features: Optional[int] = 1000
    max_df: Optional[float] = 0.95
    min_df: Optional[int] = 2
    ngram_range: Optional[tuple] = (1, 2)
    stop_words:  Optional[str] = "english"
    sample_fraction: Optional[float] = 1.0

class ModelRegistrationParams(BaseModel):
    tracking_uri: str = MLFLOW_URL
    experiment_name: str
    model_name: str
    run_id: Optional[str] = None
    tags: Optional[str] = None
    selected_index: Optional[int] = None

class ManageTagsParams(BaseModel):
    tracking_uri: Optional[str] = MLFLOW_URL
    model_name: str
    version: Optional[str] = None
    action: str
    tag_key: Optional[str] = None
    tag_value: Optional[str] = None

class DisplayArtifactsParams(BaseModel):
    run_id: str
    tracking_uri: Optional[str] = None

class Artifact(BaseModel):
    path: str
    is_dir: bool

def set_tracking_uri(tracking_uri: Optional[str] = None):
    uri = tracking_uri if tracking_uri else MLFLOW_URL
    mlflow.set_tracking_uri(uri)

def fetch_preprocessed_data(api_uri):
    # Send GET request to the API
    response = requests.get('{uri}/api/v1/preprocessed_dataset'.format(uri=api_uri))

    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Parse each line of JSON response and accumulate in a list
            data = [json.loads(line) for line in response.content.splitlines()]

            # Convert the list of dictionaries to a Pandas DataFrame
            df = pd.DataFrame(data)

            # Display the DataFrame
            return df
        except requests.exceptions.JSONDecodeError:
            print("Error: Failed to parse JSON response.")
    else:
        print(f"Error: {response.status_code}")
        


def compute_tfidf_similarity(movies, column_name='genres',  **tfidfargs):
    """
    Computes the TF-IDF matrix and cosine similarity matrix for a specified column in a DataFrame.
    
    Parameters:
    movies (pd.DataFrame): A pandas DataFrame containing the text data to be analyzed.
    column_name (str): The name of the column in the DataFrame to apply TF-IDF (default is 'genres').
    stop_words (str or list): Stop words to be used by the TfidfVectorizer (default is 'english').
    
    Returns:
    tuple: A tuple containing:
        - feature_names_with_index (list): List of tuples with feature indices and names.
        - sim_matrix (numpy.ndarray): Cosine similarity matrix computed from the TF-IDF matrix.
    """
    # Create an object for TfidfVectorizer with the specified stop words
    tfidf_vectorizer = TfidfVectorizer(**tfidfargs)
    
    # Apply the TfidfVectorizer to the specified column
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies[column_name])
    
    # Get the feature names from the TfidfVectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_names_with_index = list(enumerate(feature_names))
    
    # Compute the cosine similarity matrix from the TF-IDF matrix
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return tfidf_vectorizer, feature_names_with_index, sim_matrix

def train_model(**tfidfargs):
        
    stop_words = tfidfargs.get("stop_words", 'english')
    max_features = tfidfargs.get("max_features", 10000)
    max_df = tfidfargs.get("max_df", 1)
    min_df = tfidfargs.get("min_df", 1)
    ngram_range = tfidfargs.get("ngram_range",(1, 2))
    sample_fraction = tfidfargs.pop("sample_fraction",1 )
    
    movie_data = fetch_preprocessed_data(API_DATABASE_URL)
    movie_data = movie_data.sample(frac=sample_fraction)
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_df", max_df)
        mlflow.log_param("min_df", min_df)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("stop_words", stop_words)
        mlflow.log_param("sample_fraction", sample_fraction)

        tfidf_vectorizer,feature_names_with_index, sim_matrix = compute_tfidf_similarity(movie_data,  column_name='genres', **tfidfargs)

         # Log vocabulary size
        vocab_size = len(tfidf_vectorizer.vocabulary_)
        mlflow.log_metric("vocabulary_size", vocab_size)

        # Save the similarity matrix to a file
        np.save("sim_matrix.npy", sim_matrix)

        # Log the similarity matrix as an artifact
        mlflow.log_artifact("sim_matrix.npy")

        # Log the model
        mlflow.sklearn.log_model(tfidf_vectorizer, "model", registered_model_name="Content_Vectorizer")

@app.get("/", response_class=HTMLResponse)
def home():
    return '''
    <h1>MovieLens Recommendation API</h1>
    <p>API for training the content based filter.</p>
    '''

@app.post("/train_content_filter")
def train_content_based_filter(params: TrainingParams):
# Set the MLFlow tracking URI
    mlflow.set_tracking_uri(MLFLOW_URL)

    # Set the experiment name
    mlflow.set_experiment("Train_Contentbased_Filter")
    file = '/mlflow/api_train_content_sees_this'
    with open(file, 'w') as fp:
        pass

    try:
        train_model(**params.dict())
        return {"message": "Training finished successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def display_artifacts(client, run_id: str) -> List[dict]:
    """
    Display all artifacts in a run.

    Args:
        client: MLflow client
        run_id: ID of the run to inspect
    """
    artifacts = client.list_artifacts(run_id)
    artifacts_info = []

    for idx, artifact in enumerate(artifacts, 1):
        artifact_info = {
            "index": idx,
            "path": artifact.path,
            "type": "directory" if artifact.is_dir else "file",
            "nested": []
        }
        if artifact.is_dir:
            nested_artifacts = client.list_artifacts(run_id, artifact.path)
            for nested in nested_artifacts:
                artifact_info["nested"].append(nested.path)
        artifacts_info.append(artifact_info)
        
    return artifacts_info

def select_model_path(artifacts: List[dict], selected_index: int) -> str:
    """
    Select an artifact directory to use for model registration.

    Args:
        artifacts: List of artifacts
        selected_index: Index of the selected artifact
    Returns:
        str: Selected artifact path
    """
    # Filter only directories
    dirs = [art for art in artifacts if art["type"] == "directory"]

    if not dirs:
        raise Exception("No directories found in artifacts")

    if len(dirs) == 1:
        return dirs[0]["path"]

    if 1 <= selected_index <= len(dirs):
        return dirs[selected_index - 1]["path"]
    else:
        raise ValueError(f"Selected index {selected_index} is out of range")

def get_model_uri(client, tracking_uri: str, experiment_name: str, selected_index: int, run_id: Optional[str] = None) -> dict:
    """
    Get model URI either from a specific run_id or the latest successful run in an experiment.
    """
    set_tracking_uri(tracking_uri)

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiments = client.search_experiments()
        available_experiments = [exp.name for exp in experiments]
        raise Exception(f"Experiment '{experiment_name}' not found. Available experiments: {available_experiments}")

    if run_id:
        pass
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs.empty:
            raise Exception(f"No successful runs found in experiment '{experiment_name}'")
        run_id = runs.iloc[0].run_id

    # Get run information and artifacts
    artifacts = display_artifacts(client, run_id)
    if selected_index == -1:
        selected_index = len(artifacts)  # Default to the last artifact

    # Select model path
    model_path = select_model_path(artifacts, selected_index)
    model_uri = f"runs:/{run_id}/{model_path}"
    return {"model_uri": model_uri, "run_id":  run_id}

def register_model(client , model_uri: str, model_name: str, tags: Optional[dict] = None):
    """
    Register a model and set its tags.

    Args:
        model_uri: URI of the model to register.
        model_name: Name to register the model under.
        tags: Dictionary of tags to set (optional).
    
    Returns:
        dict: Model details including name and version.
    """

    try:
        # Register the model
        model_details = mlflow.register_model(model_uri, model_name)

        # Set tags if provided
        if tags:
            for key, value in tags.items():
                client.set_registered_model_tag(model_name, key, value)

        return {
            "model_name": model_details.name,
            "version": model_details.version
        }

    except Exception as e:
        raise Exception(f"Failed to register model: {str(e)}")


def manage_tags(model_name: str, version: Optional[str] = None, action: str = None, tag_key: str = None, tag_value: str = None):
    """
    Manage tags for a registered model or specific version
    """
    client = mlflow.tracking.MlflowClient()
    try:
        if action == "add" and tag_key and tag_value:
            if version:
                client.set_model_version_tag(model_name, version, tag_key, tag_value)
            else:
                client.set_registered_model_tag(model_name, tag_key, tag_value)
            return f"Tag {tag_key}={tag_value} set successfully"

        elif action == "delete" and tag_key:
            if version:
                client.delete_model_version_tag(model_name, version, tag_key)
            else:
                client.delete_registered_model_tag(model_name, tag_key)
            return f"Tag {tag_key} deleted successfully"

        elif action == "list":
            if version:
                model_version = client.get_model_version(model_name, version)
                tags = model_version.tags
            else:
                model = client.get_registered_model(model_name)
                tags = model.tags
            return tags

        else:
            return "Invalid action or missing parameters"

    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/register_model/")
def register_model_endpoint(params: ModelRegistrationParams):
    try:
        set_tracking_uri(params.tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Get model URI
        result = get_model_uri(
            client=client,
            tracking_uri=params.tracking_uri if params.tracking_uri else MLFLOW_URL,
            experiment_name=params.experiment_name,
            selected_index=params.selected_index if params.selected_index else -1,  # Default to last path if index is not provided
            run_id=params.run_id
        )

        # Parse initial tags if provided
        initial_tags = {}
        if params.tags:
            for tag_pair in params.tags.split(','):
                key, value = tag_pair.split('=')
                initial_tags[key.strip()] = value.strip()

        # Register model with initial tags
        model_details = register_model(
            client,
            result["model_uri"], 
            params.model_name, 
            initial_tags
            )

        return {"status": "Model registered successfully", "model_details": model_details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/manage_tags/")
def manage_tags_endpoint(params: ManageTagsParams):
    set_tracking_uri(params.tracking_uri)
    result = manage_tags(
        model_name=params.model_name,
        version=params.version,
        action=params.action,
        tag_key=params.tag_key,
        tag_value=params.tag_value
    )
    if "Error" in result:
        raise HTTPException(status_code=400, detail=result)
    return {"status": result}

@app.post("/display_artifacts/")
def display_artifacts_endpoint(params: DisplayArtifactsParams):
    try:
        set_tracking_uri(params.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        artifacts = display_artifacts(client, params.run_id)
        return artifacts
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/list_experiments/")
def list_experiments_endpoint(tracking_uri: Optional[str] = Query(None, description="Tracking URI for the MLflow server"), experiment_name: Optional[str] = Query(None, description="Name of the experiment to filter")):
    try:
        set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found.")
            experiments = [experiment]
        else:
            experiments = client.search_experiments()

        experiments_info = []
        for experiment in experiments:
            experiment_info = {
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.name,
                "runs": []
            }
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time
                }
                experiment_info["runs"].append(run_info)
            experiments_info.append(experiment_info)

        return {"experiments": experiments_info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))