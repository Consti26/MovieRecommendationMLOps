import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import pickle

# ============================================ SIDEBAR ============================================ 
st.sidebar.image("references/movie_recommender.png", use_column_width=True)
st.sidebar.title("Table of contents")
pages = [
    "Home", 
    "About the project",
    "Demonstration",  
    "Global architecture", 
    "Database", 
    "Preprocessing", 
    "Training with MLflow", 
    "Inference with MLflow", 
    "Orchestration with Airflow", 
    "Conclusion"
]
page = st.sidebar.radio("Go to", pages)

# ============================================ PAGE 0 (Home) ============================================
if page == pages[0]:
    st.markdown("""
    <div style='text-align: center; padding-top: 10vh;'>
        <h1 style='font-size: 60px;'>Movie Recommender System</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('--------------------------------------------------------------------------')
    st.markdown("<h2 style='text-align: center;'>Alexander Kramer</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Constance Fromonot</h2>", unsafe_allow_html=True)


# ============================================ PAGE 1 (About the project) ============================================
if page == pages[1]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Context & Presentation of the project</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h1 style='text-align: center; color: #e51e25; '>75%-80%</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>of the content watched on Netflix is driven by its recommendation system</h5>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center;color: #e51e25; '>35%</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>of Amazon's total revenue are driven by its recommendation engine</h5>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h1 style='text-align: center;color: #e51e25; '>150%</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>rate to which personalized recommendation systems can boost conversion</h5>", unsafe_allow_html=True)

    # Challenges
    col1, col2 = st.columns([0.5, 8.5])
    with col1:
        st.header("🗻")
    with col2:
        st.write("""##### **Challenges:** In today’s data-driven landscape, personalized experiences are key to engaging users and driving business value. \
        This project focuses on developing a recommender system embedded within an MLOps framework to ensure the solution is robust, scalable, and maintainable.\
        performance of the operations.""")

    # Objectives
    col1, col2 = st.columns([0.5, 8.5])
    with col1:
        st.header("🎯")
    with col2:
        st.write("""##### **Objectives:**""")
        st.write("""##### 1. Develop a content based recommender system.""")
        st.write("""##### 2. Establish an End-to-End MLOps Pipeline.""")

    # Content based movie recommender
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Content based movie recommender</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.write(
        """We are using the MovieLens dataset. It contains data on movie, its characteristics, and interactions with users. It can be found here :
        <a href="https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset" target="_blank">here</a>.""", 
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([6, 3])
    with col1:
        st.write("""In our case, we have chosen to develop a content based recommender system. It answer to the affirmation : """)
        st.write(""" **If you liked this movie, this one is similar to it, so you should like it.**""")
        st.write("""In this approach, we assume that users are more likely to enjoy movies with similar features to the ones they have liked in the past. \
        The features can include genres, cast, directors, and other descriptive elements of the movies. But in our case, we are going to use only the genre.""")
    with col2:
        st.image("references/content_based.png", use_column_width=True)
    st.write("""We focus on the content features, such as genres, to make personalized recommendations. This approach is particularly useful in scenarios where \
            user-item interaction data is limited, making it effective for handling the "cold start" problem for new users or items.""")

    # End-to-End MLOps Pipeline
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>End-to-End MLOps Pipeline</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.write("""The ultimate goal is to deliver a robust, scalable, and adaptive solution that continuously meets user expectations and drives business success. \
             We are going to look more in depth this part of the project.""")


# ============================================ PAGE 2 (Demonstration) ============================================
if page == pages[2]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Get a movie recommendation</h1>", unsafe_allow_html=True)

    # Current trends
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Current trends</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.image("references/trends.png", use_column_width=True)

    # Ask for a recommendation
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Ask for a recommendation </h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.write("**Give the name of a movie and we will give you our 10 recommendations based on it.**")
    
    # Insert Search Bar
    search_query = st.text_input("Enter the movie name:")
    if st.button("Search"):
        st.write(f"Searching recommendations for: {search_query}")
        # insert code of the inference api 


# ============================================ PAGE 3 (Global architecture) ============================================
if page == pages[3]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Global architecture</h1>", unsafe_allow_html=True)
    st.image("references/final_architecture.jpg", use_column_width=True)

    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Characteristics of the architecture </h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)

    # Global infos
    st.write("""
    - Each compononent is dockerized to ensure the robustness of the system.
    - The process is : 
        1. The datachecks and pre processing script query the database, perform the pre processing and write the preprocessed dataset in the database to a new path.
        2. The training script query the pre processed dataset, train a model and get a trained dataset.
        3. MLFlow log the model(s) and trained dataset.
        4. The inference script is triggered by the request of recommendation in the front page.
    - Airflow orchestrate the data checks and preprocessing script, followed by the training script.
    """)

    st.write("") # line break
    st.write("") # line break

    # Docker characteristics
    st.write("#### Docker characteristics")
    st.write("""
    The docker-compose file orchestrates 5 of the docker containers (Airflow is managed on the side).

    - **Build Context:** All docker containers use a custom Dockerfile located inside the associated folder (data, features, mlflow, models).
    - **Exposure:** Each docker container exposes a port to display API endpoints, application UIs, etc.
    - **Volumes:** Volumes are defined in `mlflow_data` for persisting MLflow data, and in `database_raw_volume` and `database_processed_volume` for storing raw and processed database files.
    - **Network:** All services are connected to a common network (`movie_recommendation_network`), facilitating seamless communication.
    """)

    # - **Dependency Management:** all dependencies are Installed via Python packages with pip.
    # - **Port Configuration:** all application accepts a build-time argument to define the application’s port, which is then exposed and used when starting the server.
    # - **Running the API:** all applications are started with Uvicorn, binding to `0.0.0.0` to allow external access.


# ============================================ PAGE 4 (Database) ============================================
if page == pages[4]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Database</h1>", unsafe_allow_html=True)
    #st.image("references/final_architecture.jpg", use_column_width=True)

    # Schema of the process
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Global overview</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.image("references/database.png", use_column_width=True)

    # Database API
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Database API</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **Data Ingestion & Storage:**
        - Downloads the MovieLens dataset from Kaggle and ingests CSV files into an SQLite database in manageable chunks (controlled by *CHUNKSIZE*), ensuring efficient memory use.
    - **API Endpoints (Using FastAPI):**
        - The database has **13 APIs endpoint**.
        - **Data Sent:** It provides endpoint to query the data from outside the database (in our case, from the preprocessing or the training script.)
        - **Data Insertion:** It provides endpoints to stream large datasets (movies and preprocessed data), or both single and batch insertion of new movie and rating records.
    """)

    # Containerization specificities
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Containerization specificities</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        - **Base Image & Setup:**  
            Uses a lightweight `python:3.9-slim` image. The working directory is set to `/home/api_database`, and necessary files (script and requirements) are copied into the container.
        - **Directory Structure:**  
            Creates directories for processed and raw data to mimic the expected local file structure.
    """)
    

# ============================================ PAGE 5 (Preprocessing) ============================================
if page == pages[5]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Pre processing</h1>", unsafe_allow_html=True)

    # Schema of the process
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Global overview</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.image("references/preprocessing.png", use_column_width=True)

    # Preprocessing Script Overview
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Preprocessing script</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **API & Environment Setup:**
        - Implements a pre-processing API to request data from the database with a Dynamic Database Connection
    - **Data Integrity Checks:**
        - Validates the DataFrame structure (checks for expected columns, unique IDs, correct data types, and non-empty genres).
    - **Data Preprocessing Steps:**
        - **Perform preprocessing on the dataset and data cleaning.**
    - **Database Table Management and data Insertion:**
        - Calls an API endpoint to create (or reset) the preprocessed_dataset table.
    """, unsafe_allow_html=True)

    # Containerization specificities
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Containerization specificities</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **Base Image & Setup:**
        - Uses a lightweight `python:3.9-slim` image. The working directory is set to `/home/api_preprocess_content/data`, and necessary files (script and requirements) are copied into the container.
    - **Directory Structure:**
        - Creates any required directories (e.g., `/home/api_preprocess_content/data`).
    """, unsafe_allow_html=True)



# ============================================ PAGE 6 (Training) ============================================
if page == pages[6]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Training phase</h1>", unsafe_allow_html=True)

    # Schema of the process
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Global overview</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.image("references/training.png", width=300)

    # Training Script
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Training script</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    

    # Containerization specificities
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #fdb94a;'>Containerization specificities</h3>
        <hr style='border: 0; height: 1px; background-color: #fdb94a; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)







# ============================================ PAGE 9 (Conclusion) ============================================
if page == pages[9]:
    st.markdown("<h1 style='text-align: center; color: #fdb94a;'>Conclusion</h1>", unsafe_allow_html=True)

    # Strength :
    # - Dynamic Database Connection in each script

    # Weaknesses : 
    # No user management to the db, streamlit etc


    # Improvment : 
    # Add user management : it would allow us to create different roles : user to only consume the recommendation, admin to load csv into the db ...
    
