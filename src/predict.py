"""
predict.py — Inférence sur de nouvelles données
Projet : Analyse Comportementale Clientèle Retail
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')


def predict_churn(input_data: dict) -> dict:
    """
    Prédit si un client va churner.
    Args:
        input_data : dict des features numériques du client (déjà encodées)
    Returns:
        {'churn': 0/1, 'probability': float, 'label': str}
    """
    model  = joblib.load(os.path.join(MODELS_DIR, 'rf_churn_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_classification.pkl'))

    df        = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)

    prediction  = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        'churn':       int(prediction),
        'probability': round(float(probability), 4),
        'label':       'Parti' if prediction == 1 else 'Fidèle',
    }


def predict_monetary(input_data: dict) -> dict:
    """
    Prédit le montant total dépensé (MonetaryTotal) par un client.
    Returns:
        {'predicted_monetary': float}
    """
    model  = joblib.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_regression.pkl'))

    df        = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    return {'predicted_monetary': round(float(prediction), 2)}


def predict_cluster(input_data: dict) -> dict:
    """
    Assigne un nouveau client à un cluster KMeans.
    Returns:
        {'cluster': int}
    """
    model  = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_cluster.pkl'))
    pca    = joblib.load(os.path.join(MODELS_DIR, 'pca_cluster.pkl'))

    df        = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)
    df_pca    = pca.transform(df_scaled)

    cluster = model.predict(df_pca)[0]
    return {'cluster': int(cluster)}


if __name__ == '__main__':
    # Exemple d'utilisation (adapter les features à votre modèle entraîné)
    sample = {
        'Recency': 30, 'Frequency': 5, 'MonetaryAvg': 240.0,
        'Age': 35.0, 'SupportTicketsCount': 1.0, 'SatisfactionScore': 4.0
    }
    print("Churn prediction :", predict_churn(sample))
