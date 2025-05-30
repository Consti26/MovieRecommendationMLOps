import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import requests
import os 

def get_data_from_api():
    """Fetch data from the database API."""
    BASE_URL = "http://localhost:8000" 
    url = f"{BASE_URL}/api/v1/movies" 
    response = requests.get(url, stream=True)  # Stream the response
    
    if response.status_code == 200:
        # Initialize an empty list to collect rows of data
        data = []
        
        # Stream the response and process each chunk
        for line in response.iter_lines(decode_unicode=True):
            if line:  # Make sure the line is not empty
                try:
                    # Parse each line as a JSON object
                    row = eval(line)  # Convert line to a dictionary (this may be a safer option than eval in some cases)
                    data.append(row)
                except Exception as e:
                    print(f"Error parsing line: {line}, Error: {e}")
        
        # Convert the collected data into a pandas DataFrame
        return pd.DataFrame(data)
    else:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")

def check_movie_data(df):
    """
    Checks if the movie DataFrame meets the expected format and constraints.
    """
    try:
        print("Checking movie DataFrame...")

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Check column names
        expected_columns = ["movieId", "title", "genres"]
        if list(df.columns) != expected_columns:
            raise ValueError(f"Columns do not match expected format: {expected_columns}")

        # Check if `movieId` is unique
        if df["movieId"].duplicated().any():
            raise ValueError("Duplicate movieId values found in the DataFrame")

        # Check if `movieId` is an integer
        if not pd.api.types.is_integer_dtype(df["movieId"]):
            raise ValueError("movieId column must contain integers only")

        # Check for missing values
        #if df.isnull().any().any():
        #    raise ValueError("The DataFrame contains missing values")

        # Check if genres are non-empty strings
        if not df["genres"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0).all():
            raise ValueError("Invalid genres found in the DataFrame")

        print("Movie DataFrame passed all checks!\n")

    except Exception as e:
        print(f"Error in movie DataFrame: {e}")

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

def drop_nan_values(movies):
    """
    Drops rows with any null values from the DataFrame.
    
    Args:
    - df (pd.DataFrame): Input DataFrame
    
    Returns:
    - pd.DataFrame: DataFrame without null values
    """
    return movies.dropna()

def main():
    # Start MLflow experiment
    mlflow.set_experiment("Content_based_preprocessing")

    with mlflow.start_run():
        try:
            # Get data from movie.csv 
            movies = get_data_from_api()

            # Check movie.csv
            if not movies.empty:
                check_movie_data(movies)
            else:
                print(f"File not found: {movies}")

            # Apply pre processing  
            movies.rename(columns={'title':'title_year'}, inplace=True)
            movies['title_year'] = movies['title_year'].apply(lambda x: x.strip()) # remove spaces in tilte_year
            movies['title'] = movies['title_year'].apply(extract_title)
            movies['year'] = movies['title_year'].apply(extract_year)
            movies = filter_movies_with_genres(movies)
            movies = clean_genre_column(movies)
            movies = drop_nan_values(movies)
            print("Preprocessed dataset has been created. Saving it into the db ..")

            # Create the table in the database
            BASE_URL="http://localhost:8000"
            CREATE_TABLE_URL = f"{BASE_URL}/api/v1/database/create_preprocessed_table"
            INSERT_DATA_URL = f"{BASE_URL}/api/v1/database/insert_preprocessed_data"
            response = requests.post(CREATE_TABLE_URL)
            if response.status_code == 200:
                print("✅ Table created successfully!")
            else:
                print(f"❌ Error creating table: {response.text}")
                exit()

            # Convert DataFrame to JSON format expected by the API
            movies_json = movies.to_dict(orient="records")  # List of dictionaries

            # Send data to API for insertion
            response = requests.post(INSERT_DATA_URL, json=movies_json)

            if response.status_code == 200:
                print("✅ Data inserted successfully!")
            else:
                print(f"❌ Error inserting data: {response.text}")

            # Log parameters
            #mlflow.log_param("Similar_movie", movie_title)
            #mlflow.log_param("number_neighbors", number_of_reco)

            # Log custom metrics 
            #mlflow.log_metric("avg_distance", np.mean(distances)) - not meaningfull

            # Log artifacts
            #recommendation.to_csv("recommendation.csv", index=False)
            #mlflow.log_artifact("recommendation.csv")

        except Exception as e:
            print(f"Error during preprocessing run: {e}")
            raise

if __name__ == "__main__":
    main()
