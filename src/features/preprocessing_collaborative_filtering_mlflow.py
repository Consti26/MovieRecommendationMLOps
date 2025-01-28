import pandas as pd
import numpy as np
import mlflow

# Get data
file_path_movie = '../raw_data/movie.csv'
file_path_rating ='../raw_data/rating.csv'
file_path_tag ='../raw_data/tag.csv'
file_path_genome_tag ='../raw_data/genome_tags.csv'
file_path_genome_score ='../raw_data/genome_scores.csv'

df_movie = pd.read_csv(file_path_movie)
df_rating = pd.read_csv(file_path_rating)
df_tag = pd.read_csv(file_path_tag)
df_genome_tag = pd.read_csv(file_path_genome_tag)
df_genome_score = pd.read_csv(file_path_genome_score)

# Create functions
def merging_datasets(df: pd.DataFrame, df_to_merge: pd.DataFrame, join_on: str ) -> pd.DataFrame:
    """merging 2 dataframes + removing movie with 0 tags with inner join

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: DataFrame merged
    """
    df = pd.merge(df, df_to_merge, on=join_on)
    print(f"MERGING DATASETS : number of distinct movieid = {df.movieId.nunique()}")
    return df

def remove_columns(df: pd.DataFrame, lst_columns_to_remove: list) -> pd.DataFrame:
    """drop selected columns

    Args:
        df (pd.DataFrame): dataframe name
        lst_columns_to_remove (list): list of columns to remove

    Returns:
        pd.DataFrame: DataFrame with selected columns
    """
    df = df.drop(columns=lst_columns_to_remove, axis=1)
    print(f"LIST OF COLUMNS KEPT : {df.columns.to_list()}")
    return df

def lowercase_str(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase strings

    Args:
        df (pd.DataFrame): Pandas DataFrame

    Returns:
        pd.DataFrame: Lowercase strings in Pandas DataFrame
    """
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    print(f"LOWERCASE EVERYTHING : number of distinct movieid = {df.movieId.nunique()}")
    return df

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """remove duplicate rows

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: DataFrame without duplicate rows
    """
    df = df.drop_duplicates(subset=df.columns.to_list()).reset_index(drop=True)
    print(f"DROP DUPLICATES ROWS : number of distinct movieid =  {df.movieId.nunique()}")
    return df


def main():
    # Start MLflow experiment
    mlflow.set_experiment("Collaborative_filtering_preprocessing")

    with mlflow.start_run():
        try:
            # Apply functions
            df = remove_columns(df_movie, ['genres'])
            df = merging_datasets(df, df_rating, 'movieId')
            df = remove_columns(df, ['timestamp'])
            df = remove_duplicate_rows(df)
            df.head(10)
            df.to_csv('../processed_data/df_collaborative_filtering.csv', sep = ',')

            # Log parameters
            #mlflow.log_param("Similar_movie", movie_title)
            #mlflow.log_param("number_neighbors", number_of_reco)

            # Log custom metrics 
            #mlflow.log_metric("avg_distance", np.mean(distances)) - not meaningfull

            # Log artifacts
            #recommendation.to_csv("recommendation.csv", index=False)
            #mlflow.log_artifact("recommendation.csv")

        except Exception as e:
            print(f"Error during collaborative filtering preprocessing run: {e}")
            raise

if __name__ == "__main__":
    main()