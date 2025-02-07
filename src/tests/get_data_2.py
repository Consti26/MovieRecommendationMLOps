import pandas as pd
import requests

def get_data_from_api():
    """Fetch data from the database API."""
    BASE_URL = "http://localhost:8000" 
    print(BASE_URL)
    url = f"{BASE_URL}/api/v1/movies" 
    print(url)
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

# Call the function to fetch and process the data
movies = get_data_from_api()
print(movies.head())