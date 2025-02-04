import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


#=================================== FUNCTIONS CHECK ON DATA PIPELINE ===================================
def check_movie_data(file_path):
    """
    Checks if the movie.csv data meets the expected format and constraints.
    """
    try:
        print(f"Checking file: {file_path}")
        df = pd.read_csv(file_path)

        # Check column names
        expected_columns = ["movieId", "title", "genres"]
        if list(df.columns) != expected_columns:
            raise ValueError(f"Columns do not match expected format: {expected_columns}")

        # Check if `movieId` is unique
        if df["movieId"].duplicated().any():
            raise ValueError("Duplicate movieId values found in movie.csv")

        # Check if `movieId` is an integer
        if not pd.api.types.is_integer_dtype(df["movieId"]):
            raise ValueError("movieId column must contain integers only")

        # Check for missing values
        #if df.isnull().any().any():
        #    raise ValueError("movie.csv contains missing values")

        # Check if genres are non-empty strings
        if not df["genres"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0).all():
            raise ValueError("Invalid genres found in movie.csv")

        print("movie.csv passed all checks!\n")

    except Exception as e:
        print(f"Error in movie.csv: {e}")


def check_rating_data(file_path):
    """
    Checks if the rating.csv data meets the expected format and constraints.
    """
    try:
        print(f"Checking file: {file_path}")
        df = pd.read_csv(file_path)

        # Check column names
        expected_columns = ["userId", "movieId", "rating", "timestamp"]
        if list(df.columns) != expected_columns:
            raise ValueError(f"Columns do not match expected format: {expected_columns}")

        # Check if `userId` and `movieId` are integers
        if not pd.api.types.is_integer_dtype(df["userId"]):
            raise ValueError("userId column must contain integers only")
        if not pd.api.types.is_integer_dtype(df["movieId"]):
            raise ValueError("movieId column must contain integers only")

        # Check if `rating` is a float and within the range 0.0 - 5.0
        #if not pd.api.types.is_float_dtype(df["rating"]):
        #    raise ValueError("rating column must contain floats only")
        if not df["rating"].between(0.0, 5.0).all():
            raise ValueError("rating column contains values outside the range 0.0 - 5.0")

        # Check if `timestamp` is a valid datetime
        try:
            pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Invalid timestamp format in rating.csv")

        # Check for missing values
        if df.isnull().any().any():
            raise ValueError("rating.csv contains missing values")

        print("rating.csv passed all checks!\n")

    except Exception as e:
        print(f"Error in rating.csv: {e}")

def check_data():
    """
    Main function to check raw data files.
    """
    raw_data_folder = '../raw_data/'
    movie_file = os.path.join(raw_data_folder, "movie.csv")
    rating_file = os.path.join(raw_data_folder, "rating.csv")

    # Check movie.csv
    if os.path.exists(movie_file):
        check_movie_data(movie_file)
    else:
        print(f"File not found: {movie_file}")

    # Check rating.csv
    if os.path.exists(rating_file):
        check_rating_data(rating_file)
    else:
        print(f"File not found: {rating_file}")

#=================================== FUNCTIONS PREPROCESSING PIPELINE CONTENT BASED ===================================

def extract_title(title):
    """
    Extracts the title of a movie, excluding the year if it exists in the title.
    
    Parameters:
    title (str): The movie title, potentially including the release year at the end in the format " (YYYY)".
    
    Returns:
    str: The cleaned title without the year, or the original title if no year is present.
    """
    # Extract the last 4 characters (assumed year) from the title.
    year = title[-5:-1]  # Slice to extract "YYYY" if present.
    
    # Check if the extracted part is numeric, indicating the presence of a year.
    if year.isnumeric():
        # Remove the year and the surrounding parentheses from the title.
        title_no_year = title[:-7]
        return title_no_year
    else:
        # Return the original title if no year is found.
        return title
    

def extract_year(title):
    """
    Extracts the release year from a movie title if it is present.
    
    Parameters:
    title (str): The movie title, potentially including the release year at the end in the format " (YYYY)".
    
    Returns:
    int or float: The extracted year as an integer if present, otherwise NaN (Not a Number).
    """
    # Extract the last 4 characters (assumed year) from the title.
    year = title[-5:-1]  # Slice to extract "YYYY" if present.
    
    # Check if the extracted part is numeric, indicating a valid year format.
    if year.isnumeric():
        return int(year)  # Convert the year to an integer and return it.
    else:
        return np.nan  # Return NaN if no valid year is found.
    

def filter_movies_with_genres(movies):
    """
    Filters out movies with '(no genres listed)' in the 'genres' column.
    
    Parameters:
    movies (pd.DataFrame): A pandas DataFrame containing a 'genres' column.
    
    Returns:
    pd.DataFrame: A new DataFrame with rows containing '(no genres listed)' removed and the index reset.
    """
    # Filter out rows where the 'genres' column has the value '(no genres listed)'.
    filtered_movies = movies[~(movies['genres'] == '(no genres listed)')]

    # Reset the index of the resulting DataFrame, dropping the old index.
    filtered_movies = filtered_movies.reset_index(drop=True)

    return filtered_movies


def clean_genre_column(movies):
    """
    Cleans and standardizes the 'genres' column in a DataFrame.
    
    - Replaces the '|' character with a space.
    - Standardizes specific genre names:
      - 'Sci-Fi' is replaced with 'SciFi'.
      - 'Film-Noir' is replaced with 'Noir'.
    
    Parameters:
    movies (pd.DataFrame): A pandas DataFrame containing a 'genres' column.
    
    Returns:
    pd.DataFrame: The modified DataFrame with cleaned and standardized 'genres' column.
    """
    # Replace all occurrences of '|' with a space in the 'genres' column.
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    # Replace 'Sci-Fi' with 'SciFi' in the 'genres' column.
    movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi', regex=False)
    
    # Replace 'Film-Noir' with 'Noir' in the 'genres' column.
    movies['genres'] = movies['genres'].str.replace('Film-Noir', 'Noir', regex=False)
    
    return movies

def content_based_preprocessing():
    # Start MLflow experiment
    mlflow.set_experiment("Content_based_preprocessing")

    with mlflow.start_run():
        try:
            # Apply functions
            movie_path = '../raw_data/movie.csv'
            movies = pd.read_csv(movie_path)
            movies.rename(columns={'title':'title_year'}, inplace=True)
            movies['title_year'] = movies['title_year'].apply(lambda x: x.strip()) # remove spaces in tilte_year
            movies['title'] = movies['title_year'].apply(extract_title)
            movies['year'] = movies['title_year'].apply(extract_year)
            movies = filter_movies_with_genres(movies)
            movies = clean_genre_column(movies)
            movies.to_csv('../processed_data/df_content_filtering.csv', sep = ',')

            # Log parameters
            #mlflow.log_param("Similar_movie", movie_title)
            #mlflow.log_param("number_neighbors", number_of_reco)

            # Log custom metrics 
            #mlflow.log_metric("avg_distance", np.mean(distances)) - not meaningfull

            # Log artifacts
            #recommendation.to_csv("recommendation.csv", index=False)
            #mlflow.log_artifact("recommendation.csv")

        except Exception as e:
            print(f"Error during content based preprocessing run: {e}")
            raise

#=================================== FUNCTIONS MODELLING PIPELINE CONTENT BASED ===================================

def compute_tfidf_similarity(movies, column_name='genres', stop_words='english'):
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
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    
    # Apply the TfidfVectorizer to the specified column
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies[column_name])
    
    # Get the feature names from the TfidfVectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_names_with_index = list(enumerate(feature_names))
    
    # Compute the cosine similarity matrix from the TF-IDF matrix
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return feature_names_with_index, sim_matrix


def get_title_year_from_index(index, movies):
    """
    Retrieves the 'title_year' value corresponding to a given index in the DataFrame.

    Parameters:
    index (int): The index for which the title and year are to be retrieved.
    movies (pd.DataFrame): The DataFrame containing the 'title_year' column.

    Returns:
    str: The 'title_year' value (e.g., "Movie Title (Year)") for the given index.
    """
    # Filter the DataFrame to match the given index and retrieve the 'title_year' value.
    return movies.loc[index, 'title_year']  # .loc is more explicit for indexing


def get_index_from_title(title, movies):
    """
    Retrieves the index of a given title in the DataFrame.

    Parameters:
    title (str): The title of the movie.
    movies (pd.DataFrame): The DataFrame containing the 'title' column.

    Returns:
    int: The index of the movie with the given title.
    """
    # Filter the DataFrame to match the given title and retrieve its index.
    return movies[movies['title'] == title].index[0]  # Access the first matching index directly


def recommend_movie(movie_title, df, number_of_reco, column_name='genres', stop_words='english'):
    """
    Recommends a movie based on the similarity of genres using TF-IDF and cosine similarity.
    
    Parameters:
    movie_title (str): The title of the movie to base the recommendation on.
    df (pd.DataFrame): The DataFrame containing the movie data with 'title', 'title_year', and 'genres' columns.
    number_of_reco (int): The number of recommandation that the use wants
    column_name (str): The column in the DataFrame to use for similarity calculation (default is 'genres').
    stop_words (str or list): Stop words to be used by the TfidfVectorizer (default is 'english').
    
    Returns:
    str: A recommendation message with the name and year of the most similar movie.
    """
    # Compute the TF-IDF similarity matrix for the 'genres' column
    sim_matrix = compute_tfidf_similarity(df, column_name, stop_words)

    # Get the index of the movie based on the title using the pre-defined function
    try:
        movie_index = get_index_from_title(movie_title, df)
    except IndexError:
        return f"The movie titled '{movie_title}' is not found in the dataset."

    list_most_similar_movie_title = []
    list_most_similar_movie_year = []
    for i in range(1,number_of_reco): 
        # Get the similarity scores for the movie
        similarity_scores = sim_matrix[1][movie_index]

        # Find the most similar movie (excluding the movie itself)
        most_similar_index = similarity_scores.argsort()[-1-i]  # Get the index of the most similar movie
        most_similar_movie_title = df.loc[most_similar_index, 'title']
        most_similar_movie_year = df.loc[most_similar_index, 'title_year']
        list_most_similar_movie_title.append(most_similar_movie_title)
        list_most_similar_movie_year.append(most_similar_movie_year)

    recommendations_df = pd.DataFrame({
        'recommended_title': list_most_similar_movie_title,
        'recommended_year': list_most_similar_movie_year
    })  
    
    return recommendations_df

def content_based_modelling():
    # Start MLflow experiment
    mlflow.set_experiment("Content_based_Experiment")

    with mlflow.start_run():
        try:
            # Apply functions
            file_path_processed_data = '../processed_data/df_content_filtering.csv'
            df = pd.read_csv(file_path_processed_data)
            movie_title = "Inception"  # Replace with the title of the movie you want to get recommendations for
            number_of_reco = 10  # Number of recommendations to return
            recommendation = recommend_movie(movie_title, df, number_of_reco)

            # Log parameters
            mlflow.log_param("Similar_movie", movie_title)
            mlflow.log_param("number_neighbors", number_of_reco)

            # Log custom metrics 
            #mlflow.log_metric("avg_distance", np.mean(distances)) - not meaningfull

            # Log artifacts
            recommendation.to_csv("recommendation.csv", index=False)
            mlflow.log_artifact("recommendation.csv")

        except Exception as e:
            print(f"Error during content based modelling run: {e}")
            raise



#=================================== DEFINE AIRFLOW DAG ===================================

# DAG configuration
default_args = {
    'owner': 'MLOps Datascientest',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    #'retries': 1,
    #'retry_delay': timedelta(minutes=2),
}

# Define DAG
movie_recommender_dag = DAG(
    dag_id='movie_recommender_dag',
    description='Make movie recommendation',
    tags=['recommendersystem', 'datascientest', 'movies'],
    schedule_interval=timedelta(minutes=30),  # Exécution toutes les demi-heures  # schedule_interval=None
    catchup=False,
    default_args=default_args
)

# Define tasks
check_data_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data,
    provide_context=True,
    dag=movie_recommender_dag,
)

content_based_preprocessing_task = PythonOperator(
    task_id='content_based_preprocessing',
    python_callable=content_based_preprocessing,
    provide_context=True,
    dag=movie_recommender_dag,
)

content_based_modelling_task = PythonOperator(
    task_id='content_based_modelling',
    python_callable=content_based_modelling,
    provide_context=True,
    dag=movie_recommender_dag,
)

check_data_task >> content_based_preprocessing_task >> content_based_modelling_task