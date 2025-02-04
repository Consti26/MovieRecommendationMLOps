import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlfow
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
MLFLOW_URI = os.getenv('MLFLOW_URI', 'http://localhost:8080')

# Set the MLFlow tracking URI
mlflow.set_tracking_uri(MLFLOW_URI)

# Set the experiment name
mlflow.set_experiment("Train_Contentbased_Filter")

app = FastAPI()

class TrainingParams(BaseModel):
    max_features: int = 1000
    max_df: float = 0.95
    min_df: int = 2
    ngram_range: tuple = (1, 2)
    stop_word:  Optional[str] = None


try:
    # DNS-Lookup durchführen
    api_address = socket.gethostbyname("02907780e411")
    print(f"Die IP-Adresse von {target_container_name} ist: {api_address}")
except socket.gaierror as e:
    print(f"Fehler beim Abrufen der IP-Adresse für {target_container_name}: {e}")

@app.get("/", response_class=HTMLResponse)
def home():
    return '''
    <h1>MovieLens Training API</h1>
    <p>API for training the content based filter.</p>
    '''

def fetch_preprocessed_data(API_name):
    try:
        # DNS-Lookup durchführen
        # api_address = socket.gethostbyname(PREPROCESSING_API)
        api_address='0.0.0.0'
        print(f"Die IP-Adresse von {PREPROCESSING_API} ist: {api_address}")
    except socket.gaierror as e:
        print(f"Fehler beim Abrufen der IP-Adresse für {target_container_name}: {e}")
     
    # Send GET request to the API
    response = requests.get('http://{address}:{port}/api/v1/movies'.format(address=api_address, port=PREPROCESSING_PORT ))

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

    movie_data = fetch_preprocessed_data(PREPROCESSING_API)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_df", max_df)
        mlflow.log_param("min_df", min_df)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("stop_words", stop_words)

        tfidf_vectorizer, sim_matrix = compute_tfidf_similarity(movie_data,  column_name='genres', **tfidfargs)

         # Log vocabulary size
        vocab_size = len(tfidf_vectorizer.vocabulary_)
        mlflow.log_metric("vocabulary_size", vocab_size)

        # Save the similarity matrix to a file
        np.save("sim_matrix.npy", sim_matrix)

        # Log the similarity matrix as an artifact
        mlflow.log_artifact("sim_matrix.npy")

        # Log the model
        mlflow.sklearn.log_model(tfidf_vectorizer, "model", registered_model_name="Content_Vectorizer")

@app.post("/train_content_filter")
def train_content_based_filter(params: TrainingParams):
    try:
        train_model(**TrainingParams)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
