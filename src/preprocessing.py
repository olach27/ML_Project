"""
preprocessing.py — Nettoyage, encodage et feature engineering
Projet : Analyse Comportementale Clientèle Retail

Pipeline complet :
  1. Suppression des features inutiles (NewsletterSubscribed → constante)
  2. Correction des valeurs aberrantes (SupportTicketsCount, SatisfactionScore)
  3. Parsing de RegistrationDate (formats mixtes UK/ISO/US)
  4. Feature engineering depuis LastLoginIP
  5. Création de nouvelles features
  6. Encodage des variables catégorielles (ordinal + one-hot + target)
  7. Imputation des valeurs manquantes (KNN)
"""

import os
import sys
import numpy as np
import pandas as pd
import ipaddress
from sklearn.impute import KNNImputer

# Permet d'importer utils.py depuis le même dossier src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_dataframe, ensure_dirs

# ── Chemins relatifs au projet ───────────────────────────────
BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH      = os.path.join(BASE, 'data', 'raw',
                              'retail_customers_COMPLETE_CATEGORICAL.csv')
PROCESSED_PATH = os.path.join(BASE, 'data', 'processed',
                               'retail_customers_processed.csv')


# ─────────────────────────────────────────────────────────────
# 1. SUPPRESSION DES FEATURES INUTILES
# ─────────────────────────────────────────────────────────────

def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes à variance nulle (valeur unique).
    → NewsletterSubscribed = toujours 'Yes' dans ce dataset.
    """
    df = df.copy()
    to_drop = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"[✓] Supprimées (variance nulle) : {to_drop}")
    else:
        print("[✓] Aucune feature à variance nulle.")
    return df


# ─────────────────────────────────────────────────────────────
# 2. CORRECTION DES VALEURS ABERRANTES
# ─────────────────────────────────────────────────────────────

def fix_aberrant_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs codées invalides par NaN :
      - SupportTicketsCount : -1 → NaN  |  999 → NaN
      - SatisfactionScore   : -1 → NaN  |  99  → NaN
      - SatisfactionScore   :  0 → NaN  (note hors échelle 1-5)
    """
    df = df.copy()

    col = 'SupportTicketsCount'
    mask = df[col].isin([-1, 999])
    df.loc[mask, col] = np.nan
    print(f"[✓] {col} : {mask.sum()} valeurs aberrantes (-1, 999) → NaN")

    col = 'SatisfactionScore'
    mask = df[col].isin([-1, 99])
    df.loc[mask, col] = np.nan
    print(f"[✓] {col} : {mask.sum()} valeurs aberrantes (-1, 99) → NaN")

    return df


# ─────────────────────────────────────────────────────────────
# 3. PARSING DE REGISTRATIONDATE
# ─────────────────────────────────────────────────────────────

def parse_registration_date(df: pd.DataFrame,
                             col: str = 'RegistrationDate') -> pd.DataFrame:
    """
    Parse la colonne date (formats mixtes : DD/MM/YY, YYYY-MM-DD, MM/DD/YYYY…)
    et extrait : RegYear, RegMonth, RegDay, RegWeekday.
    Supprime la colonne originale.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    df['RegYear']    = df[col].dt.year
    df['RegMonth']   = df[col].dt.month
    df['RegDay']     = df[col].dt.day
    df['RegWeekday'] = df[col].dt.weekday   # 0=Lundi … 6=Dimanche

    n_failed = df[col].isna().sum()
    if n_failed > 0:
        print(f"[!] {n_failed} dates non parsées → NaT (seront imputées)")

    df.drop(columns=[col], inplace=True)
    print(f"[✓] '{col}' → RegYear, RegMonth, RegDay, RegWeekday")
    return df


# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING — LASTLOGINIP
# ─────────────────────────────────────────────────────────────

def engineer_ip_features(df: pd.DataFrame,
                          col: str = 'LastLoginIP') -> pd.DataFrame:
    """
    Extrait deux features depuis l'adresse IP :
      - IP_IsPrivate  : 1 si IP privée (192.168.x.x, 10.x.x.x…), 0 sinon
      - IP_FirstOctet : premier octet (indique la classe réseau)
    Supprime la colonne originale.
    """
    df = df.copy()

    def _is_private(ip_str):
        try:
            return int(ipaddress.ip_address(str(ip_str)).is_private)
        except Exception:
            return np.nan

    def _first_octet(ip_str):
        try:
            return int(str(ip_str).split('.')[0])
        except Exception:
            return np.nan

    df['IP_IsPrivate']  = df[col].apply(_is_private)
    df['IP_FirstOctet'] = df[col].apply(_first_octet)
    df.drop(columns=[col], inplace=True)
    print(f"[✓] '{col}' → IP_IsPrivate, IP_FirstOctet")
    return df


# ─────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING — NOUVELLES FEATURES
# ─────────────────────────────────────────────────────────────

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features à partir des existantes :
      - MonetaryPerDay  : dépense moyenne par jour depuis le dernier achat
      - AvgBasketValue  : valeur moyenne du panier
      - TenureRatio     : ratio recency / ancienneté (activité récente vs durée relation)
    """
    df = df.copy()

    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = (df['MonetaryTotal']
                            / df['Frequency'].replace(0, np.nan))
    df['TenureRatio']    = (df['Recency']
                            / df['CustomerTenureDays'].replace(0, np.nan))

    print("[✓] Nouvelles features : MonetaryPerDay, AvgBasketValue, TenureRatio")
    return df


# ─────────────────────────────────────────────────────────────
# 6. ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────

# Ordre défini pour chaque feature ordinale (basé sur les valeurs réelles du dataset)
ORDINAL_MAPPINGS = {
    'AgeCategory': {
        '18-24': 0, '25-34': 1, '35-44': 2,
        '45-54': 3, '55-64': 4, '65+': 5, 'Inconnu': -1
    },
    'SpendingCategory': {
        'Low': 0, 'Medium': 1, 'High': 2, 'VIP': 3
    },
    'PreferredTimeOfDay': {
        'Matin': 0, 'Midi': 1, 'Après-midi': 2, 'Soir': 3, 'Nuit': 4
    },
    'LoyaltyLevel': {
        'Nouveau': 0, 'Jeune': 1, 'Établi': 2, 'Ancien': 3, 'Inconnu': -1
    },
    'ChurnRiskCategory': {
        'Faible': 0, 'Moyen': 1, 'Élevé': 2, 'Critique': 3
    },
    'BasketSizeCategory': {
        'Petit': 0, 'Moyen': 1, 'Grand': 2, 'Inconnu': -1
    },
}

# Features nominales → One-Hot Encoding
ONE_HOT_COLS = [
    'RFMSegment',       # Champions, Fidèles, Potentiels, Dormants
    'CustomerType',     # Hyperactif, Régulier, Occasionnel, Nouveau, Perdu
    'FavoriteSeason',   # Hiver, Printemps, Été, Automne
    'Region',           # UK, Europe du Nord/Sud/Est/Centrale, Océanie, Autre
    'WeekendPreference',# Weekend, Semaine, Inconnu
    'ProductDiversity', # Spécialisé, Modéré, Explorateur
    'Gender',           # M, F, Unknown
    'AccountStatus',    # Active, Suspended, Pending, Closed
]


def encode_categoricals(df: pd.DataFrame,
                         target_col: str = 'Churn') -> pd.DataFrame:
    """
    Encode toutes les variables catégorielles :
      - Ordinal encoding  pour les features ordonnées (ORDINAL_MAPPINGS)
      - One-Hot encoding  pour les features nominales (ONE_HOT_COLS)
      - Target encoding   pour Country (haute cardinalité → moyenne de Churn)
    """
    df = df.copy()

    # --- Ordinal ---
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            print(f"[✓] Ordinal encoded : {col}")

    # --- One-Hot ---
    existing_ohe = [c for c in ONE_HOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_ohe, drop_first=False, dtype=int)
    print(f"[✓] One-Hot encoded  : {existing_ohe}")

    # --- Target encoding pour Country ---
    if 'Country' in df.columns and target_col in df.columns:
        country_target_mean = df.groupby('Country')[target_col].mean()
        df['Country_encoded'] = df['Country'].map(country_target_mean)
        df.drop(columns=['Country'], inplace=True)
        print(f"[✓] Target encoded   : Country → Country_encoded")

    return df


# ─────────────────────────────────────────────────────────────
# 7. IMPUTATION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────────────────────

def impute_missing(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   n_neighbors: int = 5) -> tuple:
    """
    KNNImputer appliqué uniquement sur les features numériques.
    Fit sur X_train → transform sur X_test (pas de data leakage).
    Retourne (X_train_imputed, X_test_imputed, imputer).
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=n_neighbors)

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = imputer.transform(X_test[numeric_cols])

    print(f"[✓] KNNImputer (k={n_neighbors}) sur {len(numeric_cols)} features numériques.")
    return X_train, X_test, imputer


# ─────────────────────────────────────────────────────────────
# 8. PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────

def full_preprocessing_pipeline(raw_path: str = RAW_PATH,
                                  processed_path: str = PROCESSED_PATH
                                  ) -> pd.DataFrame:
    """
    Exécute toute la chaîne de prétraitement dans l'ordre correct
    et sauvegarde le résultat dans data/processed/.
    """
    print("\n" + "=" * 55)
    print("  PIPELINE PRÉTRAITEMENT — DÉMARRAGE")
    print("=" * 55)

    df = pd.read_csv(raw_path)
    print(f"[→] Données brutes chargées : {df.shape}")

    df = drop_useless_features(df)
    df = fix_aberrant_values(df)
    df = parse_registration_date(df)
    df = engineer_ip_features(df)
    df = create_new_features(df)
    df = encode_categoricals(df)

    ensure_dirs(os.path.dirname(processed_path))
    save_dataframe(df, processed_path)

    print("=" * 55)
    print(f"[✓] Pipeline terminé. Shape final : {df.shape}")
    print("=" * 55)
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    full_preprocessing_pipeline()
