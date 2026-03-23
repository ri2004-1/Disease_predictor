# Diabetes Risk Predictor

A machine learning web application that predicts diabetes risk 
based on patient health data.

## What it does
- Takes 8 patient health inputs via interactive sliders
- Uses Random Forest ML model trained on 768 patient records
- Predicts High Risk or Low Risk of diabetes
- Achieves 78% accuracy on test data

## Technologies used
- Python 3.11
- pandas — data handling
- scikit-learn — Random Forest ML model
- Streamlit — web app interface

## Dataset
PIMA Indians Diabetes Database from Kaggle — 768 patient records

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## Skills demonstrated
Data preprocessing, ML model training, web deployment
