"""
train_model.py — Entraînement des modèles ML
Projet : Analyse Comportementale Clientèle Retail

Modèles :
  1. Clustering     → KMeans avec PCA 2D  (segmentation clients)
  2. Classification → Random Forest       (prédiction Churn)
  3. Régression     → Linear Regression  (prédiction MonetaryTotal)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score, silhouette_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import scale_features, save_dataframe, ensure_dirs

# ── Chemins ──────────────────────────────────────────────────
BASE            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH  = os.path.join(BASE, 'data', 'processed',
                                'retail_customers_processed.csv')
TRAIN_TEST_DIR  = os.path.join(BASE, 'data', 'train_test')
MODELS_DIR      = os.path.join(BASE, 'models')
REPORTS_DIR     = os.path.join(BASE, 'reports')


# ─────────────────────────────────────────────────────────────
# 1. SÉPARATION TRAIN / TEST
# ─────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame,
                   target: str,
                   test_size: float = 0.2,
                   random_state: int = 42) -> tuple:
    """
    Sépare X/y en train/test (stratifié pour la classification).
    Sauvegarde les 4 fichiers CSV dans data/train_test/.
    ⚠ La variable cible (y) n'est JAMAIS normalisée.
    """
    ensure_dirs(TRAIN_TEST_DIR)

    X = df.drop(columns=[target])
    y = df[target]

    # Garder uniquement les colonnes numériques (sécurité)
    X = X.select_dtypes(include=[np.number])

    # Stratification uniquement pour classification (peu de classes)
    stratify = y if y.nunique() <= 20 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    save_dataframe(X_train,          os.path.join(TRAIN_TEST_DIR, 'X_train.csv'))
    save_dataframe(X_test,           os.path.join(TRAIN_TEST_DIR, 'X_test.csv'))
    save_dataframe(y_train.to_frame(), os.path.join(TRAIN_TEST_DIR, 'y_train.csv'))
    save_dataframe(y_test.to_frame(),  os.path.join(TRAIN_TEST_DIR, 'y_test.csv'))

    print(f"[✓] Train : {X_train.shape}  |  Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# 2. CLUSTERING — KMeans
# ─────────────────────────────────────────────────────────────

def train_clustering(X: pd.DataFrame,
                     n_clusters: int = 4) -> dict:
    """
    KMeans sur les données réduites par PCA (2 composantes).
    Évalue avec le score silhouette.
    Sauvegarde : kmeans_model.pkl, scaler_cluster.pkl, pca_cluster.pkl
    """
    ensure_dirs(MODELS_DIR, REPORTS_DIR)

    X_num = X.select_dtypes(include=[np.number])

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)
    print(f"[✓] KMeans — {n_clusters} clusters  |  Silhouette : {score:.4f}")

    # Visualisation
    plt.figure(figsize=(8, 6))
    for k in range(n_clusters):
        mask = labels == k
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {k}', s=10)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='X', s=200, c='black', label='Centroïdes')
    plt.title(f'KMeans Clustering (k={n_clusters}) — PCA 2D')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'clustering_pca.png'), dpi=150)
    plt.show()

    # Sauvegarde
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_cluster.pkl'))
    joblib.dump(pca,    os.path.join(MODELS_DIR, 'pca_cluster.pkl'))
    print(f"[✓] Modèles cluster sauvegardés dans {MODELS_DIR}/")

    return {'model': kmeans, 'labels': labels,
            'silhouette': score, 'scaler': scaler, 'pca': pca}


# ─────────────────────────────────────────────────────────────
# 3. CLASSIFICATION — Random Forest (Churn)
# ─────────────────────────────────────────────────────────────

def train_classification(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: pd.Series,
                          y_test: pd.Series) -> dict:
    """
    Random Forest avec GridSearchCV pour prédire le Churn.
    class_weight='balanced' pour compenser le déséquilibre des classes.
    Sauvegarde : rf_churn_model.pkl, scaler_classification.pkl
    """
    ensure_dirs(MODELS_DIR, REPORTS_DIR)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train, X_test, scaler = scale_features(X_train, X_test, numeric_cols)

    param_grid = {
        'n_estimators':      [100, 200],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5],
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf, param_grid, cv=5,
                        scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    print(f"\n[✓] Meilleurs hyperparamètres : {grid.best_params_}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred,
                                 target_names=['Fidèle (0)', 'Parti (1)']))

    # Matrice de confusion
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Fidèle', 'Parti'],
        cmap='Blues', ax=ax
    )
    ax.set_title('Matrice de Confusion — Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix_churn.png'), dpi=150)
    plt.show()

    # Feature importance (top 15)
    importances = pd.Series(best.feature_importances_,
                             index=X_train.columns).nlargest(15)
    plt.figure(figsize=(9, 6))
    importances.sort_values().plot(kind='barh', color='steelblue')
    plt.title('Top 15 Feature Importances — Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance_rf.png'), dpi=150)
    plt.show()

    joblib.dump(best,   os.path.join(MODELS_DIR, 'rf_churn_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_classification.pkl'))
    print(f"[✓] Modèle classification sauvegardé dans {MODELS_DIR}/")

    return {'model': best, 'y_pred': y_pred,
            'scaler': scaler, 'best_params': grid.best_params_}


# ─────────────────────────────────────────────────────────────
# 4. RÉGRESSION — LinearRegression (MonetaryTotal)
# ─────────────────────────────────────────────────────────────

def train_regression(X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: pd.Series,
                      y_test: pd.Series) -> dict:
    """
    Régression linéaire pour prédire MonetaryTotal.
    ⚠ y (target) n'est PAS normalisée.
    Sauvegarde : linear_regression_model.pkl, scaler_regression.pkl
    """
    ensure_dirs(MODELS_DIR, REPORTS_DIR)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train, X_test, scaler = scale_features(X_train, X_test, numeric_cols)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"[✓] Régression Linéaire  |  RMSE : {rmse:.2f}  |  R² : {r2:.4f}")

    # Réel vs prédit
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', label='Parfait')
    plt.xlabel('Valeur réelle (£)')
    plt.ylabel('Valeur prédite (£)')
    plt.title('Régression — Réel vs Prédit (MonetaryTotal)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'regression_actual_vs_pred.png'), dpi=150)
    plt.show()

    joblib.dump(model,  os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_regression.pkl'))
    print(f"[✓] Modèle régression sauvegardé dans {MODELS_DIR}/")

    return {'model': model, 'y_pred': y_pred,
            'rmse': rmse, 'r2': r2, 'scaler': scaler}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = pd.read_csv(PROCESSED_PATH)
    print(f"[→] Données chargées : {df.shape}")

    # ── Classification : Churn ─────────────────────────────
    print("\n=== CLASSIFICATION (Churn) ===")
    X_train, X_test, y_train, y_test = split_and_save(df, target='Churn')
    train_classification(X_train, X_test, y_train, y_test)

    # ── Régression : MonetaryTotal ─────────────────────────
    print("\n=== RÉGRESSION (MonetaryTotal) ===")
    X_r, X_r_test, y_r, y_r_test = split_and_save(df, target='MonetaryTotal')
    train_regression(X_r, X_r_test, y_r, y_r_test)

    # ── Clustering ─────────────────────────────────────────
    print("\n=== CLUSTERING (KMeans) ===")
    X_cluster = df.select_dtypes(include=[np.number]).drop(
        columns=['Churn'], errors='ignore')
    train_clustering(X_cluster, n_clusters=4)
