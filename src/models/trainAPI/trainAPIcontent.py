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
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# Get the paths from environment variables
PREPROCESSING_API = os.getenv('PREPROCESSING_API', 'db_api')
PREPROCESSING_PORT = int(os.getenv('PREPROCESSING_PORT', '8000'))
MLFLOW_CONTAINER = os.getenv('MLFLOW_CONTAINER', 'mlflow')
MLFLOW_PORT = int(os.getenv('MLFLOW_PORT', '5000'))

try:
    # DNS-Lookup durchführen
    preprocessing_address = socket.gethostbyname(PREPROCESSING_API)
    PREPROCESSING_URI = 'http://{address}:{port}'.format(address=preprocessing_address, port=PREPROCESSING_PORT )
    print(f"Die URI von {PREPROCESSING_API} ist: {PREPROCESSING_URI}")
except socket.gaierror as e:
    PREPROCESSING_URI = 'http://localhost:8000'
    print(f"Fehler beim Abrufen der IP-Adresse für {PREPROCESSING_API}: {e}")

try:
    # DNS-Lookup durchführen MLFlow
    mlflow_address = socket.gethostbyname(MLFLOW_CONTAINER)
    MLFLOW_URI = 'http://{address}:{port}'.format(address=mlflow_address, port=MLFLOW_PORT )
    print(f"Die URI von {MLFLOW_CONTAINER} ist: {MLFLOW_URI}")
except socket.gaierror as e:
    MLFLOW_URI = 'http://localhost:5000'
    print(f"Fehler beim Abrufen der IP-Adresse für {MLFLOW_CONTAINER}: {e}")


app = FastAPI()

class TrainingParams(BaseModel):
    max_features: Optional[int] = 1000
    max_df: Optional[float] = 0.95
    min_df: Optional[int] = 2
    ngram_range: Optional[tuple] = (1, 2)
    stop_words:  Optional[str] = None
    sample_fraction: Optional[float] = 1.0

def fetch_preprocessed_data(api_uri):
    # Send GET request to the API
    response = requests.get('{uri}/api/v1/movies'.format(uri=api_uri))

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
    
    movie_data = fetch_preprocessed_data(PREPROCESSING_URI)
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
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Set the experiment name
    mlflow.set_experiment("Train_Contentbased_Filter")

    try:
        train_model(**params.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))