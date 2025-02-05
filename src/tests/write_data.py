import requests
import pandas as pd

# Define API endpoints
BASE_URL="http://localhost:8000"
CREATE_TABLE_URL = f"{BASE_URL}/api/v1/database/create_preprocessed_table"
INSERT_DATA_URL = f"{BASE_URL}/api/v1/database/insert_preprocessed_data"

# Sample DataFrame
data = {
    "movieId": [1, 2, 3, 4, 5],
    "title_year": [
        "Toy Story (1995)",
        "Jumanji (1995)",
        "Grumpier Old Men (1995)",
        "Waiting to Exhale (1995)",
        "Father of the Bride Part II (1995)"
    ],
    "genres": [
        "Adventure Animation Children Comedy Fantasy",
        "Adventure Children Fantasy",
        "Comedy Romance",
        "Comedy Drama Romance",
        "Comedy"
    ],
    "title": [
        "Toy Story",
        "Jumanji",
        "Grumpier Old Men",
        "Waiting to Exhale",
        "Father of the Bride Part II"
    ],
    "year": [1995, 1995, 1995, 1995, 1995]
}

df = pd.DataFrame(data)

# 1. Create the table in the database
response = requests.post(CREATE_TABLE_URL)
if response.status_code == 200:
    print("✅ Table created successfully!")
else:
    print(f"❌ Error creating table: {response.text}")
    exit()

# 2. Convert DataFrame to JSON format expected by the API
movies_json = df.to_dict(orient="records")  # List of dictionaries

# 3. Send data to API for insertion
response = requests.post(INSERT_DATA_URL, json=movies_json)

if response.status_code == 200:
    print("✅ Data inserted successfully!")
else:
    print(f"❌ Error inserting data: {response.text}")


def get_preprocessed_data_from_api():
    """Fetch data from the database API."""
    BASE_URL = "http://localhost:8000" 
    url = f"{BASE_URL}/api/v1/preprocessed_dataset"
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
    
df = get_preprocessed_data_from_api()
print(df.head())