# Retail Customer Churn Prediction

A Python project for customer behavior analysis and churn prediction in retail.
It includes a Flask dashboard, trained models, dataset splits, and analysis notebooks.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the Web App](#run-the-web-app)
- [Run the Notebooks](#run-the-notebooks)
- [Data and Models](#data-and-models)
- [Key Files](#key-files)
- [Troubleshooting](#troubleshooting)

## Project Overview

This project is built to demonstrate a full machine learning workflow for retail customer churn prediction.
It provides:

- a Flask app for online churn scoring and customer segmentation,
- trained classification and clustering models,
- processed datasets for training and testing,
- notebooks for exploration, preprocessing, PCA, modeling, and evaluation.

## Features

- Customer churn prediction using a Random Forest model.
- Customer segmentation using KMeans clustering.
- Interactive Flask interface for prediction and visualization.
- Data preprocessing and training utilities in Python scripts.
- Jupyter notebooks for reproducibility and analysis.

## Repository Structure

- `app/`
  - Flask application code (`app.py`)
  - HTML templates for pages and forms
- `data/`
  - `raw/` � original dataset files
  - `processed/` � cleaned and transformed data
  - `train_test/` � training and testing splits used by the app
- `models/`
  - serialized model files loaded by the Flask app
- `notebooks/`
  - notebooks for data exploration, preprocessing, PCA, modeling, and evaluation
- `src/`
  - helper scripts for preprocessing, training, prediction, and utilities
- `requirements.txt`
  - Python dependency list for the project
- `README.md`
  - this documentation file

## Prerequisites

- Windows
- Python 3.11 or 3.12
- `pip` (included with Python)
- Optional: a code editor such as VS Code

## Setup

From the repository root folder `ML_Project`:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Important: run these commands from `ML_Project`, because `requirements.txt` is inside `ML_Project`.

## Run the Web App

1. Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

2. Start the Flask app:

```powershell
cd app
python app.py
```

3. Open your browser and go to:

```text
http://127.0.0.1:5000
```

## Run the Notebooks

To open the data science notebooks, activate the environment then launch Jupyter Lab:

```powershell
.\venv\Scripts\activate
jupyter lab
```

The notebooks are located in `notebooks/` and cover:

- data exploration
- preprocessing
- PCA transformation
- model training
- model evaluation

## Data and Models

- `data/train_test/X_train.csv` and `data/train_test/X_test.csv` are the main feature datasets used by the app.
- `data/train_test/y_train.csv` and `data/train_test/y_test.csv` are the churn labels.
- Models in `models/` include serialized sklearn objects such as the churn Random Forest and KMeans clustering models.
- The app uses `joblib` to load saved models at startup.

## Key Files

- `app/app.py` � Flask application logic and prediction routes
- `app/templates/` � HTML templates for the web UI
- `src/preprocessing.py` � preprocessing functions for dataset preparation
- `src/train_model.py` � training scripts for models
- `src/predict.py` � prediction utilities used in the project
- `src/utils.py` � helper functions used across scripts

## Troubleshooting

- If `requirements.txt` is missing or not found, make sure you are in the `ML_Project` folder.
- If the app fails to start, confirm that `ML_Project/models` contains the required `.pkl` files.
- If model loading fails, check that the virtual environment is active and dependencies are installed.

## Next Steps

- Use the Flask UI to test customer churn predictions.
- Review the notebooks to understand how the models were built.
- Update the data preprocessing or model training scripts if you want to improve the model.