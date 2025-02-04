from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import sqlite3
import pandas as pd
from typing import List, Optional
import os
from dotenv import load_dotenv
import kagglehub
import shutil
from pydantic import BaseModel
import json

from typing import List


# Load environment variables from a .env file
load_dotenv()

# Get the paths from environment variables
DATABASE_PATH = os.getenv('DATABASE_PATH', './processed_data/movielens.db')
DATASET_PATH = os.getenv('DATASET_PATH', './raw_data')
KAGGLE_PATH = os.getenv('KAGGLE_PATH', 'grouplens/movielens-20m-dataset')
CHUNKSIZE = int(os.getenv('CHUNKSIZE', 10000))

app = FastAPI()

class Movie(BaseModel):
    movieId: int
    title: str
    genres: str

class Rating(BaseModel):
    userId: int
    movieId: int
    rating: float
    timestamp: int

class PreprocessedMovie(BaseModel):
    movieId: int
    title_year: str
    genres: str
    title: str
    year: int

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[cursor.description[idx][0]] = row[idx]
    return d

def create_db_from_csv(dataset_path: str, db_path: str, remove_existing: bool = False):
    """Creates SQLite database from CSV files with optional removal of existing database."""
    
    # Ensure the dataset_path directory exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Dataset path {dataset_path} created.")
    
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset", force_download=True)

    if os.path.isdir(dataset_path):
        shutil.copytree(path, dataset_path, dirs_exist_ok=True )
    else:
        shutil.move(path, dataset_path)

    # Ensure the parent directory of db_path exists
    db_parent_dir = os.path.dirname(db_path)
    if not os.path.exists(db_parent_dir):
        os.makedirs(db_parent_dir)
        print(f"Database parent directory {db_parent_dir} created.")

    # Check if the database file exists and remove it if needed
    if remove_existing and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Existing database {db_path} removed.")
    
    conn = sqlite3.connect(db_path)
    
    # Process each CSV file in chunks
    chunksize = CHUNKSIZE  # Adjust the chunk size as needed
    
    file_mappings = [
        ('tag.csv', 'tags'),
        ('rating.csv', 'ratings'),
        ('movie.csv', 'movies'),
        ('link.csv', 'links'),
        ('genome_scores.csv', 'genome_scores'),
        ('genome_tags.csv', 'genome_tags')
    ]

    for file_name, table_name in file_mappings:
        file_path = os.path.join(dataset_path, file_name)
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
    
    conn.close()

@app.get("/", response_class=HTMLResponse)
def home():
    return '''
    <h1>MovieLens Dataset API</h1>
    <p>API for accessing the MovieLens dataset with chunked processing.</p>
    '''

def stream_data(db_path: str, query: str, params: List = None):
    """Streams data in chunks from the SQLite database."""
    conn = sqlite3.connect(db_path)
    params = params or []
    chunksize = CHUNKSIZE

    for chunk in pd.read_sql_query(query, conn, params=params, chunksize=chunksize):
        for row in chunk.to_dict(orient='records'):
            yield json.dumps(row) + "\n"
    conn.close()

@app.get("/api/v1/movies", response_class=StreamingResponse)
def read_movies():
    """Endpoint to fetch all movies."""
    query = "SELECT * FROM movies"
    #return StreamingResponse(stream_data(DATABASE_PATH, query), media_type="application/json")
    return JSONResponse(content=movies)

@app.get("/api/v1/ratings", response_class=StreamingResponse)
def read_ratings():
    """Endpoint to fetch all ratings."""
    query = "SELECT * FROM ratings"
    return StreamingResponse(stream_data(DATABASE_PATH, query), media_type="application/json")

@app.get("/api/v1/tags", response_class=StreamingResponse)
def read_tags():
    """Endpoint to fetch all tags."""
    query = "SELECT * FROM tags"
    return StreamingResponse(stream_data(DATABASE_PATH, query), media_type="application/json")

@app.get("/api/v1/movies/filter", response_class=StreamingResponse)
def filter_movies(movieid: Optional[int] = Query(None),
                  title: Optional[str] = Query(None),
                  genres: Optional[str] = Query(None)):
    """Endpoint to filter movies based on query parameters."""
    query = "SELECT * FROM movies WHERE"
    to_filter = []
    if movieid:
        query += ' movieId=? AND'
        to_filter.append(movieid)
    if title:
        query += ' title LIKE ? AND'
        to_filter.append(f'%{title}%')
    if genres:
        query += ' genres LIKE ? AND'
        to_filter.append(f'%{genres}%')
    
    query = query.rstrip("AND") + ";" if to_filter else query.replace("WHERE", "") + ";"
    return StreamingResponse(stream_data(DATABASE_PATH, query, to_filter), media_type="application/json")

@app.get("/api/v1/ratings/filter", response_class=StreamingResponse)
def filter_ratings(userid: Optional[int] = Query(None),
                   movieid: Optional[int] = Query(None),
                   rating: Optional[float] = Query(None)):
    """Endpoint to filter ratings based on query parameters."""
    query = "SELECT * FROM ratings WHERE"
    to_filter = []
    if userid:
        query += ' userId=? AND'
        to_filter.append(userid)
    if movieid:
        query += ' movieId=? AND'
        to_filter.append(movieid)
    if rating:
        query += ' rating=? AND'
        to_filter.append(rating)
    
    query = query.rstrip("AND") + ";" if to_filter else query.replace("WHERE", "") + ";"
    return StreamingResponse(stream_data(DATABASE_PATH, query, to_filter), media_type="application/json")

@app.post("/api/v1/database/create")
def create_database(remove_existing: bool = Query(False)):
    """Endpoint to trigger the creation of the database."""
    try:
        create_db_from_csv(DATASET_PATH, DATABASE_PATH, remove_existing)
        return {"message": "Database created successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/movies")
def add_movie(movie: Movie):
    """Endpoint to add a new movie."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", 
                       (movie.movieId, movie.title, movie.genres))
        conn.commit()
        conn.close()
        return {"message": "Movie added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ratings")
def add_rating(rating: Rating):
    """Endpoint to add a new rating."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ratings (userId, movieId, rating, timestamp) VALUES (?, ?, ?, ?)", 
                       (rating.userId, rating.movieId, rating.rating, rating.timestamp))
        conn.commit()
        conn.close()
        return {"message": "Rating added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/movies/batch")
def add_movies_batch(movies: List[Movie]):
    """Endpoint to add multiple movies."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        for movie in movies:
            cursor.execute("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", 
                           (movie.movieId, movie.title, movie.genres))
        conn.commit()
        conn.close()
        return {"message": "Movies added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ratings/batch")
def add_ratings_batch(ratings: List[Rating]):
    """Endpoint to add multiple ratings."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        for rating in ratings:
            cursor.execute("INSERT INTO ratings (userId, movieId, rating, timestamp) VALUES (?, ?, ?, ?)", 
                           (rating.userId, rating.movieId, rating.rating, rating.timestamp))
        conn.commit()
        conn.close()
        return {"message": "Ratings added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/database/create_preprocessed_table")
def create_preprocessed_table():
    """Creates (or recreates) the 'preprocessed_dataset' table with specified columns."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Drop table if it already exists
        cursor.execute("DROP TABLE IF EXISTS preprocessed_dataset")

        # Create new table with required columns
        cursor.execute("""
            CREATE TABLE preprocessed_dataset (
                movieId INTEGER PRIMARY KEY,
                title_year TEXT,
                genres TEXT,
                title TEXT,
                year INTEGER
            )
        """)

        conn.commit()
        conn.close()
        return {"message": "Table 'preprocessed_dataset' created successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/database/insert_preprocessed_data")
def insert_preprocessed_data(data: List[PreprocessedMovie]):
    """Receives a list of preprocessed movie records and inserts them into the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Insert data into the table
        cursor.executemany("""
            INSERT INTO preprocessed_dataset (movieId, title_year, genres, title, year)
            VALUES (?, ?, ?, ?, ?)
        """, [(d.movieId, d.title_year, d.genres, d.title, d.year) for d in data])

        conn.commit()
        conn.close()
        return {"message": "Data inserted into 'preprocessed_dataset' successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))