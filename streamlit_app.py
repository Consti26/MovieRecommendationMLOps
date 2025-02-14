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
    "Demonstration", 
    "About the project", 
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

# ============================================ PAGE 1 (Demonstration) ============================================
if page == pages[1]:
    st.markdown("<h1 style='text-align: center; color: #1d8479;'>Context & Presentation of the project</h1>", unsafe_allow_html=True)
    # st.image(r"/Users/admin/Documents/Formation ML Engineer/Projet SNCF/Streamlit/carteidf.png", use_column_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h1 style='text-align: center;'>6 200</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>trains</h4>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center;'>2.3 million</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>validations per day</h4>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h1 style='text-align: center;'>365/365</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>days</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 8.5])
    with col1:
        st.header("🗻")
    with col2:
        st.write("""##### **Challenges:** Increasing number of passengers are frequent and it impacts the current capabilities of the \
        infrastructure. Better anticipating this increase will help SNCF offer more appropriate services and improve the \
        performance of the operations.""")

    col1, col2 = st.columns([0.5, 8.5])
    with col1:
        st.header("🎯")
    with col2:
        st.write("""##### **Objectives:** Predict the number of validations per day and per station, for the coming year.""")

    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #1d8479;'>Technical translation</h3>
        <hr style='border: 0; height: 1px; background-color: #1d8479; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.write("""Our problem is **time series prediction**, which falls under **supervised forecasting**. Forecasting involves predicting 
    future values based on previously observed data, and it is widely used to anticipate trends, behaviors, or outcomes in various domains.""")

# ============================================ PAGE 2 (About the project) ============================================
if page == pages[2]:
    st.markdown("<h1 style='text-align: center; color: #1d8479;'>Presentation of the dataset</h1>", unsafe_allow_html=True)
    st.write(
        """We have used the data given by the SNCF for a data challenge. They can be found 
        <a href="https://challengedata.ens.fr/challenges/149" target="_blank">here</a>.""", 
        unsafe_allow_html=True
    )
    st.write("The dataset looks like this:")

    st.write("##### Characteristics of the dataset")
    st.write("""
    - Series length: from 1st January 2015 to 31st December 2022 (2922 days).
    - No null values in the dataset.
    """)

    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #1d8479;'>Some visualizations</h3>
        <hr style='border: 0; height: 1px; background-color: #1d8479; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)

# ============================================ PAGE 5 (Preprocessing) ============================================
if page == pages[5]:
    st.markdown("<h1 style='text-align: center; color: #1d8479;'>Pre processing</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #1d8479;'>Pivoting the table</h3>
        <hr style='border: 0; height: 1px; background-color: #1d8479; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)
    st.write("""The original dataset has 1 row per day per station, but the data needs to be structured such that the time series 
    appear in columns side-by-side (rather than on top of one another). Now, our dataset looks like that:""")

    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='display: inline; color: #1d8479;'>Missing values and outliers handling</h3>
        <hr style='border: 0; height: 1px; background-color: #1d8479; margin-top: 10px; width: 50%;'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: left;'>
        <h5 style='display: inline; color: #1d8479;'>&#8226; Missing values</h5>
    </div>
    """, unsafe_allow_html=True)
    st.write("As mentioned previously, the original dataset had no apparent missing values:")

    st.markdown("""
    <div style='text-align: left;'>
        <h5 style='display: inline; color: #1d8479;'>&#8226; Outliers</h5>
    </div>
    """, unsafe_allow_html=True)
    st.write("""It is insufficient to flag the outliers based purely on identifying the NaN values. We should also consider 
    the stations with zero values, or values that have extreme drops (e.g. a station going from 20’000 to 5).""")
    st.write("""To identify them, we use a rolling median method by defining a threshold and window size variable. 
    This method is designed to flag these sudden drops, and output a mask identifying them. This mask is then used to set 
    all maintenance day values to NaN, in order to allow imputation methods. Later the threshold and window size can 
    be optimized as new hyperparameters.""")

    st.markdown("""
    <div style='text-align: left;'>
        <h5 style='display: inline; color: #1d8479;'>&#8226; Imputations</h5>
    </div>
    """, unsafe_allow_html=True)
    st.write("""Once these days are identified, we will impute values in place to help the model train. The approach is different for the test and training sets.""")
    st.write("""
    - Training set: imputing 0 on those values (tested against other methods such as linear, quadratic, spline and akima, Piecewise Cubic Hermite Interpolating Polynomial - PCHIP).
    - Test set: These unpredictable days will certainly have an impact on the overall MAPE (discussed further in the modeling chapter). 
      So these values do not need to be imputed.
    """)