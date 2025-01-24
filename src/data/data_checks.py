import pandas as pd
import os


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


def main():
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


if __name__ == "__main__":
    main()
