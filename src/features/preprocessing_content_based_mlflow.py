import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


