import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import mlflow
import mlflow.sklearn
from sklearn.neighbors import NearestNeighbors
from mlflow import MlflowClient

# Define functions
def matrix_dataset(df: pd.DataFrame, index: str, columns: str) :
      """Pivot and convert dataframe into matrix

    Args:
        df (pd.DataFrame): dataframe name
        index (str) : name of columns of dataframe used for index (choose "title" for item-based recommender, "userId" for user-based recommender)
        columns (str) : name of columns of dataframe used for columns (choose "userId" for item-based recommender, "title" for user-based recommender)

    Returns:
        df_pivoted (pd.DataFrame): dataframe name
        df_matrix (np.float): matrix name
    """

      # Pivot the DataFrame
      df_pivoted = df.pivot_table(index=index, columns=columns, values='rating').fillna(0)
      matrix = csr_matrix(df_pivoted.values)
      df_pivoted = pd.DataFrame(df_pivoted)

      return df_pivoted, matrix


def knn_model(df: pd.DataFrame, matrix: np.matrix, number_neighbors : int):
    """Building the model: using cosine distance and brute-force search

    Args:
        df (pd.DataFrame): dataframe pivoted name
        matrix (np.matrix): matrix name
        number_neighbors (int): number of neighbors. Count - 1 neighbor (eg: n=5 --> 4 neighbors)
        experiment_name (str): MLflow experiment name

    Returns:
        distances (np.array): array name
        indices (np.array): array name
    """

    # Instantiate the KNN model
    model_knn= NearestNeighbors(metric= 'cosine', algorithm='brute')
    model_knn.fit(matrix)

    distances = []
    indices = []

    for i in range(len(df)):

        # Predict neighbors for each row in the DataFrame
        neighbors = model_knn.kneighbors(df.iloc[i, :].values.reshape(1, -1), n_neighbors=number_neighbors)
        distances.append(neighbors[0])
        indices.append(neighbors[1])

    return distances, indices


def reshape_3d_array(arr: np.array) :
    """Reshaping 3d array into

    Args:
        arr (np.array): array name

    Returns:
        array_in_df (pd.DataFrame): dataframe name
    """

    # Initializing reshaping the 3d array into 2d array
    array_reshape = []

    # Reshaping the 3d array into 2d array using iteration
    for temp in arr:
        for elem in temp:
            array_reshape.append(elem)

    # Storing result in Dataframe
    array_in_df = pd.DataFrame(array_reshape)

    return array_in_df


# Start MLflow experiment
mlflow.set_experiment("Collaborative_Filtering_Experiment")

with mlflow.start_run():
    try:
        # Apply functions
        file_path_processed_data = '../processed_data/df_collaborative_filtering.csv'
        df = pd.read_csv(file_path_processed_data)
        df_pivoted, matrix = matrix_dataset(df, "title", "userId")
        distances, indices = knn_model(df_pivoted, matrix, 5)
        distance_df = reshape_3d_array(distances)
        indices_df = reshape_3d_array(indices)

        # Log parameters
        mlflow.log_param("number_neighbors", 5)

        # Log custom metrics 
        #mlflow.log_metric("avg_distance", np.mean(distances)) - not meaningfull

        # Log artifacts
        distance_df.to_csv("distances.csv", index=False)
        indices_df.to_csv("indices.csv", index=False)
        mlflow.log_artifact("distances.csv")
        mlflow.log_artifact("indices.csv")

    except Exception as e:
        print(f"Error during collaborative filtering modelling run: {e}")
        raise