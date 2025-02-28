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
from tfidf_vectorizer_model import TfidfVectorizerModel
from mlflow.models import infer_signature

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

class Artifact(BaseModel):
    path: str
    is_dir: bool

class DisplayArtifactsParams(BaseModel):
    run_id: str

class ModelParams(BaseModel):
    experiment_name: str
    model_name: str
    run_id: Optional[str] = None
    tags: Optional[str] = None
    selected_index: Optional[int] = None

class ManageTagsParams(BaseModel):
    model_name: str
    version: Optional[str] = None
    action: str
    tag_key: Optional[str] = None
    tag_value: Optional[str] = None



class TFIDFArgs(BaseModel):
    stop_words: Optional[str] = 'english'
    max_features: Optional[int] = 10000
    max_df: Optional[float] = 1.0
    min_df: Optional[int] = 1
    ngram_range: Optional[tuple] = (1, 2)
    

class TrainingLabels(BaseModel):
    experiment_name: Optional[str] = "Train_Contentbased_Filter"
    model_name: Optional[str] = "contentbased_filter"

class TrainingParams(BaseModel):
    labels: TrainingLabels
    tfidf_args: TFIDFArgs
    sample_fraction: Optional[float] = 0.1


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


    
class MLFlowAPI:
    def __init__(self, tracking_uri: str = MLFLOW_URL, database_uri: str = "http://your-api-database-url"):
        self.tracking_uri = tracking_uri
        self.database_uri = database_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    
    def list_experiments(self, experiment_name: Optional[str] = None):
        try:
            if experiment_name:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found.")
                experiments = [experiment]
            else:
                experiments = self.client.search_experiments()

            experiments_info = []
            for experiment in experiments:
                experiment_info = {
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment.name,
                    "runs": []
                }
                runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])
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
    
    def display_artifacts(self, params: DisplayArtifactsParams):
        try:
            artifacts = self.client.list_artifacts(params.run_id)
            artifacts_info = []

            for idx, artifact in enumerate(artifacts, 1):
                artifact_info = {
                    "index": idx,
                    "path": artifact.path,
                    "type": "directory" if artifact.is_dir else "file",
                    "nested": []
                }
                if artifact.is_dir:
                    nested_artifacts = self.client.list_artifacts(params.run_id, artifact.path)
                    for nested in nested_artifacts:
                        artifact_info["nested"].append(nested.path)
                artifacts_info.append(artifact_info)

            return artifacts_info
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    def get_model_uri(self, experiment_name: str, selected_index: int, run_id: Optional[str] = None):
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiments = self.client.search_experiments()
                available_experiments = [exp.name for exp in experiments]
                raise Exception(f"Experiment '{experiment_name}' not found. Available experiments: {available_experiments}")

            if not run_id:
                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="status = 'FINISHED'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if runs.empty:
                    raise Exception(f"No successful runs found in experiment '{experiment_name}'")
                run_id = runs.iloc[0].run_id

            artifacts = self.display_artifacts(DisplayArtifactsParams(run_id=run_id))
            if selected_index == -1:
                selected_index = len(artifacts)

            model_path = self.select_model_path(artifacts, selected_index)
            model_uri = f"runs:/{run_id}/{model_path}"
            return {"model_uri": model_uri, "run_id": run_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    def select_model_path(self, artifacts: List[dict], selected_index: int) -> str:
        dirs = [art for art in artifacts if art["type"] == "directory"]

        if not dirs:
            raise Exception("No directories found in artifacts")

        if len(dirs) == 1:
            return dirs[0]["path"]

        if 1 <= selected_index <= len(dirs):
            return dirs[selected_index - 1]["path"]
        else:
            raise ValueError(f"Selected index {selected_index} is out of range")
    
    def register_model(self, params: ModelParams):
        try:
            result = self.get_model_uri(params.experiment_name, params.selected_index if params.selected_index else -1, params.run_id)

            initial_tags = {}
            if params.tags:
                for tag_pair in params.tags.split(','):
                    key, value = tag_pair.split('=')
                    initial_tags[key.strip()] = value.strip()

            model_details = mlflow.register_model(result["model_uri"], params.model_name)
            model_version = model_details.version

            if initial_tags:
                for key, value in initial_tags.items():
                    self.client.set_model_version_tag(params.model_name, model_version, key, value)

            return {"status": "Model registered successfully", "model_details": {"name": model_details.name, "version": model_details.version}}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    def manage_tags(self, params: ManageTagsParams):
        try:
            if params.action == "add" and params.tag_key and params.tag_value:
                if params.version:
                    self.client.set_model_version_tag(params.model_name, params.version, params.tag_key, params.tag_value)
                else:
                    self.client.set_registered_model_tag(params.model_name, params.tag_key, params.tag_value)
                return f"Tag {params.tag_key}={params.tag_value} set successfully"

            elif params.action == "delete" and params.tag_key:
                if params.version:
                    self.client.delete_model_version_tag(params.model_name, params.version, params.tag_key)
                else:
                    self.client.delete_registered_model_tag(params.model_name, params.tag_key)
                return f"Tag {params.tag_key} deleted successfully"

            elif params.action == "list":
                if params.version:
                    model_version = self.client.get_model_version(params.model_name, params.version)
                    tags = model_version.tags
                else:
                    model = self.client.get_registered_model(params.model_name)
                    tags = model.tags
                return tags

            else:
                return "Invalid action or missing parameters"
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    def fetch_preprocessed_data(self):
        # Send GET request to the API
        response = requests.get('{uri}/api/v1/preprocessed_dataset'.format(uri=self.database_uri))

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

    def train_model(self, training_params: TrainingParams):
        experiment_name = training_params.labels.experiment_name
        model_name = training_params.labels.model_name
        tfidfargs = training_params.tfidf_args

        mlflow.set_experiment(experiment_name)
        try:
            stop_words = tfidfargs.stop_words
            max_features = tfidfargs.max_features
            max_df = tfidfargs.max_df
            min_df = tfidfargs.min_df
            ngram_range = tfidfargs.ngram_range
            sample_fraction = training_params.sample_fraction
            
            movie_data = self.fetch_preprocessed_data()
            movie_data = movie_data.sample(frac=sample_fraction)
            
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_param("max_features", max_features)
                mlflow.log_param("max_df", max_df)
                mlflow.log_param("min_df", min_df)
                mlflow.log_param("ngram_range", ngram_range)
                mlflow.log_param("stop_words", stop_words)
                mlflow.log_param("sample_fraction", sample_fraction)

                # Compute TF-IDF and similarity matrix
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    max_df=max_df,
                    min_df=min_df,
                    ngram_range=ngram_range,
                    stop_words=stop_words
                )
                tfidf_vectorizer_model = TfidfVectorizerModel(tfidf_vectorizer)
                tfidf_vectorizer_model.fit(movie_data['genres'])
                # tfidf_matrix = tfidf_vectorizer_model.predict(movie_data['genres'])
                # sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
                # feature_names_with_index = list(enumerate(tfidf_vectorizer.get_feature_names_out()))
                # input_example = {
                #     "model_input": "Action Adventure"
                # }
                                              
                input_signature = infer_signature(model_input=np.array(["Action Adventure"]), params={"number_of_recommendations":10})
                print(input_signature)
                # Log vocabulary size
                vocab_size = len(tfidf_vectorizer.vocabulary_)
                mlflow.log_metric("vocabulary_size", vocab_size)

                # # Save the similarity matrix and feature names to files
                # np.save("sim_matrix.npy", sim_matrix)
                # np.save("feature_names_with_index.npy", feature_names_with_index)

                # # Log the similarity matrix, feature names, and movie titles as artifacts
                # mlflow.log_artifact("sim_matrix.npy")
                # mlflow.log_artifact("feature_names_with_index.npy")
                
                # Save the movie titles
                movie_data[['title', "genres"]].to_csv("movie_data.csv", index=False , sep=',')
                mlflow.log_artifact("movie_data.csv")

                # Log the custom TfidfVectorizer model
                mlflow.pyfunc.log_model(
                    artifact_path="model", 
                    python_model=tfidf_vectorizer_model, 
                    registered_model_name=model_name,
                    signature=input_signature
                    )

                return {"message": "Training finished successfully!",
                        "signature": input_signature}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

# Create an instance of the API class
mlflow_api = MLFlowAPI(tracking_uri=MLFLOW_URL, database_uri=API_DATABASE_URL)

# FastAPI app
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return '''
    <h1>MovieLens Recommendation API</h1>
    <p>API for training the content based filter.</p>
    '''

@app.get("/list_experiments/")
def list_experiments_endpoint(experiment_name: Optional[str] = Query(None, description="Name of the experiment to filter")):
    return mlflow_api.list_experiments(experiment_name)

@app.post("/display_artifacts/")
def display_artifacts_endpoint(params: DisplayArtifactsParams):
    return mlflow_api.display_artifacts(params)

@app.post("/register_model/")
def register_model_endpoint(params: ModelParams):
    return mlflow_api.register_model(params)

@app.post("/manage_tags/")
def manage_tags_endpoint(params: ManageTagsParams):
    return mlflow_api.manage_tags(params)

@app.post("/train_content_filter")
def train_content_based_filter(params: TrainingParams):
    # Set the MLFlow tracking URI
    mlflow_api.train_model(params)