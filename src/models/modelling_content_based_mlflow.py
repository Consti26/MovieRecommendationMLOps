import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

print(recommendation)