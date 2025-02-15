from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
import socket
from typing import Optional
from tfidf_vectorizer_model import TfidfVectorizerModel

MLFLOW_CONTAINER = os.getenv('MLFLOW_CONTAINER', 'mlflow_container')
MLFLOW_PORT = int(os.getenv('MLFLOW_PORT', '5000'))

try:
    # DNS-Lookup durchführen MLFlow
    mlflow_address = socket.gethostbyname(MLFLOW_CONTAINER)
    MLFLOW_URL = 'http://{address}:{port}'.format(address=mlflow_address, port=MLFLOW_PORT )
    print(f"Die URI von {MLFLOW_CONTAINER} ist: {MLFLOW_URL}")
except socket.gaierror as e:
    MLFLOW_URL = 'http://localhost:5000'
    print(f"Fehler beim Abrufen der IP-Adresse für {MLFLOW_CONTAINER}: {e}")

class RecommendParams(BaseModel):
    movie_title: str
    number_of_recommendations: int
    genre: Optional[str] = None

class FetchNewModelParams(BaseModel):
    model_name: Optional[str] = "contentbased_filter"
    stage: Optional[str] = "production"

class MLFlowRecommendation:
    def __init__(self, model_name: str, tracking_uri: str, stage: str = "production"):
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.stage = stage
        self.model = None
        self.run_id = None
        self.sim_matrix = None
        self.feature_names_with_index = None
        self.movie_data = None
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        self.load_model_by_stage()

    def load_model_by_stage(self):
        """
        Load the model and artifacts from MLflow using the stage tag.
        """
        model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        
        for version in model_versions:
            # model_version_info = self.client.get_model_version(self.model_name, version.version)
            tags = version.tags
            if tags.get("stage") == self.stage:
                model_uri = f"models:/{self.model_name}/{version.version}" 
                try:
                    self.model = mlflow.pyfunc.load_model(model_uri)
                except mlflow.exceptions.MlflowException as e:
                    self.model = None
                    return {"warning": f"Failed to load model: {e}"}
                self.run_id = version.run_id
                # self.sim_matrix = self.load_artifact("sim_matrix.npy")
                # self.feature_names_with_index = self.load_artifact("feature_names_with_index.npy")
                artifact_uri = f"runs:/{self.run_id}/movie_data.csv"
                local_path = mlflow.artifacts.download_artifacts(artifact_uri= artifact_uri)
                self.movie_data = pd.read_csv(local_path)

                return {"status": "Model loaded successfully"}
        
        self.model = None
        return {"warning": f"Model with stage '{self.stage}' for {self.model_name} not found"}

    def load_artifact(self, artifact_path: str):
        """
        Load an artifact from MLflow.
        """
        artifact_uri = self.client.get_run(self.run_id).info.artifact_uri
        local_path = mlflow.artifacts.download_artifacts(artifact_uri, artifact_path)
        return np.load(local_path, allow_pickle=True)

    def load_movie_titles(self):
        """
        Load the movie titles and years used for the model from the artifact.
        """
        artifact_uri = self.client.get_run(self.run_id).info.artifact_uri
        local_path = mlflow.artifacts.download_artifacts(artifact_uri, "movie_titles.csv")
        df = pd.read_csv(local_path)
        return df

    def get_index_from_title(self, movie_title: str) -> int:
        """
        Get the index of the movie based on the title.
        """
        indices = self.movie_data[self.movie_data['title'] == movie_title].index.tolist()
        if not indices:
            raise HTTPException(status_code=404, detail=f"Movie titled '{movie_title}' not found")
        return indices[0]

     # Preprocessing function for genres
    def preprocess_genres(self, genres: str) -> str:
        genres = genres.replace('|', ' ')
        genres = genres.replace('Sci-Fi', 'SciFi')
        genres = genres.replace('Film-Noir', 'Noir')
        return genres
    
    def recommend_movie(self, movie_title: str, number_of_recommendations: int, genre: Optional[str] = None) -> pd.DataFrame:
        """
        Recommends a movie based on the similarity of genres.
        """
        # Ensure the model is loaded

        print(f"Movie title: {movie_title}")
        if not self.model:
            raise HTTPException(status_code=400, detail="No model is currently loaded")

        print("Model is loaded")
        # Try to get the movie genres for the given title
        movie_genres = self.movie_data.loc[self.movie_data['title'] == movie_title, 'genres']

        if movie_genres.empty:
            if genre:
                # Use the provided genre if the movie title is not found
                movie_genres = np.array([self.preprocess_genres(genre)])
            else:
                raise HTTPException(status_code=404, detail=f"Movie titled '{movie_title}' not found and no genre provided")
        else:
            movie_genres = np.array([self.preprocess_genres(movie_genres.values[0])])

        try:
            # Prepare the input for the model
            # model_input = {"model_input": movie_genres}
            # Generate recommendations using the model
            
            top_similar_indices, top_similarities = self.model.predict(movie_genres, params={"number_of_recommendations":number_of_recommendations})
            # Retrieve the recommended movie titles, years, and similarity scores
            recommended_movies = self.movie_data.iloc[top_similar_indices[0],:].copy()
            
            # print(top_similarities.flatten())
            # print(recommended_movies)
            # print(f"The shape of the DataFrame is: {recommended_movies.shape}")

            recommended_movies['similarity_score'] = top_similarities.flatten()
            recommendations_df = recommended_movies[['title', 'similarity_score']]
            
        except Exception as e:
            print(f"An error occurred during recommendation: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        return recommendations_df

    def fetch_new_model(self, model_name: Optional[str] = None, stage: Optional[str] = "production"):
        """
        Fetch a new version of the model from MLflow based on the given stage tag.
        """
        if model_name:
            self.model_name = model_name
        if stage:
            self.stage = stage
        return self.load_model_by_stage()

# Initialize the FastAPI app and the recommendation system
app = FastAPI()
recommendation_system = MLFlowRecommendation("contentbased_filter", MLFLOW_URL)

@app.post("/recommend_movie/")
def recommend_movie(params: RecommendParams):
    """
    Recommends a movie based on the similarity of genres using the production model from MLflow.
    You can specify multiple genres by separating them with whitespaces.
    It is recommended to specify at least one genre, in case that the title is not contained in the data base.
    """
    try:
        recommendations_df = recommendation_system.recommend_movie(
            params.movie_title, 
            params.number_of_recommendations, 
            params.genre
        )
        return recommendations_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/fetch_new_model/")
def fetch_new_model(params: FetchNewModelParams):
    """
    Fetch a new version of the model from MLflow based on the given stage tag.
    """
    try:
        return recommendation_system.fetch_new_model(params.model_name, params.stage)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))