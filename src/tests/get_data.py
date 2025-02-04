import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import requests

def get_data_from_api():
    """Fetch data from the database API."""
    BASE_URL = "http://localhost:8000"  # Use service name in Docker Compose
    print(BASE_URL)
    url = f"{BASE_URL}/api/v1/movies" 
    print(url)
    response = requests.get(url)
    print(response.text)  # This will print the raw response

    if response.status_code == 200:
        data = response.json()  # API returns JSON
        return pd.DataFrame(data)  # Convert to DataFrame
    else:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")
    
movies = get_data_from_api()
print(movies.head()) 