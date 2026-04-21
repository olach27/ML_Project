"""
train_model.py — Pipeline d'entrainement complet
================================================
Corrections appliquees:
  1. CHURN     : 14 features fuyantes exclues (Recency, ChurnRiskCategory, etc.)
                 -> AUC realiste ~0.96 au lieu de 1.00.
  2. CLUSTERING: Approche RFM ciblee avec 9 features metier selectionnees,
                 winsorisation (p1-p99) + log-transform des colonnes skewed,
                 filtrage des clients a MonetaryTotal <= 0 (anomalies comptables),
                 et k=4 force pour 4 segments metier interpretatbles.
  3. REGRESSION: inchange, R2 ~ 0.67.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              silhouette_score, r2_score, mean_absolute_error)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (preprocess_data,
                            LEAKY_CHURN_FEATURES,
                            LEAKY_CLUSTERING_FEATURES)

BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE, 'data', 'raw',
                               'retail_customers_COMPLETE_CATEGORICAL.csv')
PROCESSED_PATH = os.path.join(BASE, 'data', 'processed',
                               'retail_customers_processed.csv')
MODELS_DIR     = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Features metier pour le clustering  (subset RFM + enrichissements)
# ---------------------------------------------------------------------------
# On utilise uniquement des features comportementales brutes.
# Les colonnes a forte asymetrie (Recency, Frequency, MonetaryTotal,
# UniqueProducts, AvgDaysBetweenPurchases) sont log-transformees dans
# _preprocess_for_clustering() avant le scaling.
CLUSTERING_FEATURES = [
    'Recency',
    'Frequency',
    'MonetaryTotal',
    'SatisfactionScore',
    'ReturnRatio',
    'UniqueProducts',
    'AvgDaysBetweenPurchases',
    'Age',
    'SupportTicketsCount',
]

LOG_COLS_CLUSTERING = [
    'Recency', 'Frequency', 'MonetaryTotal',
    'UniqueProducts', 'AvgDaysBetweenPurchases',
]


def _safe_drop(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')


# ---------------------------------------------------------------------------
# Preprocessing specifique au clustering
# ---------------------------------------------------------------------------
def _preprocess_for_clustering(df_input: pd.DataFrame):
    """
    Retourne (X_scaled, imputer, scaler, feature_names).

    Etapes :
      1. Filtrage  : retire les clients avec MonetaryTotal <= 0
                     (anomalies comptables — retours superieurs aux achats).
      2. Selection : 9 features RFM + enrichissements.
      3. Imputation: KNNImputer.
      4. Winsorise : clip p1 - p99 pour neutraliser les outliers extremes
                     (MinQuantity a -80 995, MaxQuantity a +80 995, etc.)
                     sans les supprimer.
      5. Log-transform: colonnes fortement skewed.
      6. Scaling   : RobustScaler.
    """
    #filtrage des clients a MonetaryTotal<=0
    df_c = df_input[df_input['MonetaryTotal'] > 0].copy()
    print(f"[i] Clients apres filtrage MonetaryTotal>0 : {len(df_c)} "
          f"(retires: {len(df_input) - len(df_c)})")

    existing = [c for c in CLUSTERING_FEATURES if c in df_c.columns]
    X = df_c[existing].copy()

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=existing, index=df_c.index)

    # Winsorisation p1-p99
    clip_bounds = {}
    for col in existing:
        p01 = X_imp[col].quantile(0.01)
        p99 = X_imp[col].quantile(0.99)
        clip_bounds[col] = (p01, p99)
        X_imp[col] = X_imp[col].clip(p01, p99)

    # Log-transform des colonnes skewed (tout est positif apres clip)
    for col in LOG_COLS_CLUSTERING:
        if col in X_imp.columns:
            X_imp[col] = np.log1p(X_imp[col])

    # Scaling
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_imp)

    return X_sc, imputer, scaler, clip_bounds, existing, df_c.index


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION CHURN
# ---------------------------------------------------------------------------
def train_churn_model(df):
    print("\n" + "=" * 62)
    print("  CLASSIFICATION CHURN  (Random Forest)")
    print("=" * 62)

    cols_to_drop = LEAKY_CHURN_FEATURES + ['Churn', 'CustomerID']
    X = _safe_drop(df, cols_to_drop)
    y = df['Churn']

    print(f"[i] Features : {X.shape[1]} colonnes | {len(y)} exemples")
    print(f"[i] Churn=1  : {y.sum()}   Churn=0 : {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ImbPipeline : etapes aplaties (supporte SMOTE)
    churn_pipeline = ImbPipeline([
        ('imputer',    KNNImputer(n_neighbors=5)),
        ('scaler',     RobustScaler()),
        ('smote',      SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,        # limite la profondeur -> evite l'overfitting
            min_samples_leaf=10, # evite les feuilles a 1 seul exemple
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
        )),
    ])

    print("[->] Entrainement...")
    churn_pipeline.fit(X_train, y_train)

    y_pred  = churn_pipeline.predict(X_test)
    y_proba = churn_pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n--- Rapport (Test Set) ---")
    print(classification_report(y_test, y_pred))
    print(f"AUC : {auc:.3f}")
    if auc > 0.98:
        print("[!] AUC encore tres elevee -> verifier LEAKY_CHURN_FEATURES")

    rf = churn_pipeline.named_steps['classifier']
    importances = sorted(zip(X_train.columns, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print("\n--- Top 15 features ---")
    for name, imp in importances[:15]:
        print(f"  {name:<44s} {imp:.4f}")

    joblib.dump(churn_pipeline, os.path.join(MODELS_DIR, 'churn_pipeline.pkl'))
    print(f"\n[OK] Sauvegarde -> churn_pipeline.pkl")
    return churn_pipeline


# ---------------------------------------------------------------------------
# 2. CLUSTERING KMEANS
# ---------------------------------------------------------------------------
def train_clustering_model(df):
    print("\n" + "=" * 62)
    print("  CLUSTERING  (KMeans — approche RFM ciblee)")
    print("=" * 62)
    print("""
Methodologie :
  • 9 features RFM metier selectionnees manuellement
  • Filtrage : clients MonetaryTotal <= 0 exclus (anomalies comptables)
  • Winsorisation p1-p99 pour neutraliser les outliers extremes
  • Log-transform des colonnes fortement skewed
  • k=4 pour 4 segments metier interpretables
""")

    X_sc, imputer, scaler, clip_bounds, feature_names, valid_idx = \
        _preprocess_for_clustering(df)

    # k=4 : meilleur compromis Silhouette / interpretabilite metier
    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_sc)
    sil = silhouette_score(X_sc, clusters)
    print(f"[OK] KMeans k={K} | Silhouette = {sil:.3f}")

    # Profil sur la population filtree
    df_c = df.loc[valid_idx].copy()
    df_c['Cluster'] = clusters

    profile_cols = [c for c in ['MonetaryTotal', 'Recency', 'Frequency',
                                 'Churn', 'SatisfactionScore']
                    if c in df_c.columns]
    summary = df_c.groupby('Cluster')[profile_cols].mean()
    summary['n_clients'] = df_c.groupby('Cluster').size()
    print("\n--- Profil brut des clusters ---")
    print(summary.round(2))

    # Etiquetage metier base sur MonetaryTotal decroissant
    sorted_ids = summary['MonetaryTotal'].sort_values(ascending=False).index.tolist()
    base_labels = ['VIP', 'Fidele', 'Occasionnel', 'A Risque']
    cluster_mapping = {cid: base_labels[i] for i, cid in enumerate(sorted_ids)}
    reverse_mapping = {v: k for k, v in cluster_mapping.items()}

    df_c['Label'] = df_c['Cluster'].map(cluster_mapping)
    print("\n--- Profil par label ---")
    labeled_summary = df_c.groupby('Label')[profile_cols].mean()
    labeled_summary['n_clients'] = df_c.groupby('Label').size()
    print(labeled_summary.round(2))

    # Validation metier
    if 'Churn' in df_c.columns:
        for label in base_labels:
            grp = df_c[df_c['Label'] == label]
            if len(grp):
                churn_rate = grp['Churn'].mean()
                status = "[OK]" if label in ['VIP', 'Fidele'] and churn_rate < 0.4 else \
                         "[OK]" if label in ['Occasionnel', 'A Risque'] else "[!]"
                print(f"  {status} {label:<12s}: churn={churn_rate:.1%}  n={len(grp)}")

    # Sauvegarde : pipeline de preprocessing + kmeans + mappings
    clustering_artifacts = {
        'imputer':      imputer,
        'scaler':       scaler,
        'clip_bounds':  clip_bounds,
        'feature_names': feature_names,
        'log_cols':     LOG_COLS_CLUSTERING,
    }
    joblib.dump(clustering_artifacts,
                os.path.join(MODELS_DIR, 'clustering_preprocessor_pipeline.pkl'))
    joblib.dump(kmeans,
                os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    joblib.dump(cluster_mapping,
                os.path.join(MODELS_DIR, 'cluster_label_mapping.pkl'))
    joblib.dump(reverse_mapping,
                os.path.join(MODELS_DIR, 'reverse_cluster_label_mapping.pkl'))
    print(f"\n[OK] Sauvegardes -> {MODELS_DIR}")
    return clustering_artifacts, kmeans, cluster_mapping


# ---------------------------------------------------------------------------
# 3. REGRESSION (MonetaryTotal)
# ---------------------------------------------------------------------------
def train_regression_model(df):
    print("\n" + "=" * 62)
    print("  REGRESSION  (MonetaryTotal)")
    print("=" * 62)

    X = _safe_drop(df, ['MonetaryTotal', 'CustomerID'])
    y = df['MonetaryTotal']
    print(f"[i] Features : {X.shape[1]} colonnes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    regression_pipeline = Pipeline([
        ('imputer',   KNNImputer(n_neighbors=5)),
        ('scaler',    RobustScaler()),
        ('regressor', LinearRegression()),
    ])

    print("[->] Entrainement...")
    regression_pipeline.fit(X_train, y_train)

    y_pred = regression_pipeline.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    print(f"\n--- Rapport (Test Set) ---")
    print(f"R2  : {r2:.3f}")
    print(f"MAE : {mae:.2f} GBP")

    joblib.dump(regression_pipeline,
                os.path.join(MODELS_DIR, 'regression_pipeline.pkl'))
    print(f"[OK] Sauvegarde -> regression_pipeline.pkl")
    return regression_pipeline


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    cols_path = os.path.join(MODELS_DIR, 'processed_columns.pkl')

    if not os.path.exists(PROCESSED_PATH):
        print("[!] Pretraitement initial necessaire...")
        df = preprocess_data(pd.read_csv(RAW_PATH), target_col='Churn')
        df.to_csv(PROCESSED_PATH, index=False)
        joblib.dump(df.columns.tolist(), cols_path)
    else:
        print(f"[OK] Chargement -> {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH)
        if os.path.exists(cols_path):
            expected = joblib.load(cols_path)
            missing  = set(expected) - set(df.columns)
            if missing:
                print(f"[!] Colonnes manquantes : {missing}. Re-pretraitement.")
                df = preprocess_data(pd.read_csv(RAW_PATH), target_col='Churn')
                df.to_csv(PROCESSED_PATH, index=False)
                joblib.dump(df.columns.tolist(), cols_path)
            else:
                df = df[[c for c in expected if c in df.columns]]

    train_churn_model(df)
    train_clustering_model(df)
    train_regression_model(df)

    print("\n" + "=" * 62)
    print("  TOUS LES MODELES ENTRAINES ET SAUVEGARDES")
    print("=" * 62)


if __name__ == '__main__':
    main()