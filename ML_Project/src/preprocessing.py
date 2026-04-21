import pandas as pd
import numpy as np
import ipaddress
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# Mappings pour l'encodage ordinal
ORDINAL_MAPPINGS = {
    'AgeCategory': {
        '18-24': 0, '25-34': 1, '35-44': 2,
        '45-54': 3, '55-64': 4, '65+': 5, 'Inconnu': -1
    },
    'SpendingCategory': {
        'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3, 'Inconnu': -1
    },
    'LoyaltyLevel': {
        'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3, 'Inconnu': -1
    },
    'ChurnRiskCategory': {
        'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3, 'Inconnu': -1
    },
    'BasketSizeCategory': {
        'Small': 0, 'Medium': 1, 'Large': 2, 'Very Large': 3, 'Inconnu': -1
    },
    'PreferredTimeOfDay': {
        'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3, 'Inconnu': -1
    },
}

# Colonnes pour le One-Hot Encoding
ONE_HOT_COLS = [
    'RFMSegment', 'CustomerType', 'FavoriteSeason', 'Region',
    'WeekendPreference', 'ProductDiversity', 'Gender', 'AccountStatus',
]

# --- Fonctions de Nettoyage et Feature Engineering ---

def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    to_drop = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"[OK] Supprimees (variance nulle) : {to_drop}")
    else:
        print("[OK] Aucune feature a variance nulle.")
    return df

def fix_aberrant_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = 'SupportTicketsCount'
    mask = df[col].isin([-1, 999])
    df.loc[mask, col] = np.nan
    col = 'SatisfactionScore'
    mask = df[col].isin([-1, 99])
    df.loc[mask, col] = np.nan
    print("[OK] Valeurs aberrantes corrigees.")
    return df

def parse_registration_date(df: pd.DataFrame,
                             col: str = 'RegistrationDate') -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    df['RegYear']    = df[col].dt.year
    df['RegMonth']   = df[col].dt.month
    df['RegDay']     = df[col].dt.day
    df['RegWeekday'] = df[col].dt.weekday
    n_failed = df[col].isna().sum()
    if n_failed > 0:
        print(f"[!] {n_failed} dates non parsees -> NaT")
    df.drop(columns=[col], inplace=True)
    print(f"[OK] '{col}' -> RegYear, RegMonth, RegDay, RegWeekday")
    return df

def engineer_ip_features(df: pd.DataFrame,
                          col: str = 'LastLoginIP') -> pd.DataFrame:
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
    print(f"[OK] '{col}' -> IP_IsPrivate, IP_FirstOctet")
    return df

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cree uniquement AvgBasketValue.
    NOTE: MonetaryPerDay et TenureRatio ont ete retires car ils sont
    des derives de Recency / CustomerTenureDays, deux features hautement
    correlees au Churn (fuite de donnees).
    """
    df = df.copy()
    df['AvgBasketValue'] = (df['MonetaryTotal']
                            / df['Frequency'].replace(0, np.nan))
    print("[OK] Nouvelles features : AvgBasketValue")
    return df

def encode_categorical_features(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            print(f"[OK] '{col}' encodee ordinalement.")

    existing_ohe = [c for c in ONE_HOT_COLS if c in df.columns]
    if existing_ohe:
        df = pd.get_dummies(df, columns=existing_ohe, drop_first=False, dtype=int)
        print(f"[OK] {existing_ohe} encodees One-Hot.")

    if 'Country' in df.columns and target_col and target_col in df.columns:
        country_target_mean = df.groupby('Country')[target_col].mean()
        df['Country_encoded'] = df['Country'].map(country_target_mean)
        df.drop(columns=['Country'], inplace=True)
        print(f"[OK] Target encoded : Country -> Country_encoded")
    elif 'Country' in df.columns:
        df.drop(columns=['Country'], inplace=True)
        print("[!] 'Country' supprimee (pas de target_col).")

    return df

def impute_missing(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   n_neighbors: int = 5) -> tuple:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train_imputed = X_train.copy()
    X_test_imputed  = X_test.copy()
    X_train_imputed[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test_imputed[numeric_cols]  = imputer.transform(X_test[numeric_cols])
    print(f"[OK] Imputation KNN effectuee avec {n_neighbors} voisins.")
    return X_train_imputed, X_test_imputed, imputer

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols]  = scaler.transform(X_test[numeric_cols])
    print("[OK] Features numeriques mises a l'echelle avec RobustScaler.")
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(df: pd.DataFrame, target_col: str = None, drop_constant: bool = True) -> pd.DataFrame:
    df_processed = df.copy()
    if drop_constant and len(df_processed) > 1:
        df_processed = drop_useless_features(df_processed)
    df_processed = fix_aberrant_values(df_processed)
    df_processed = parse_registration_date(df_processed)
    df_processed = engineer_ip_features(df_processed)
    df_processed = create_new_features(df_processed)
    df_processed = encode_categorical_features(df_processed, target_col)
    print("[OK] Pretraitement initial termine.")
    return df_processed


# =============================================================================
# LISTES DES FEATURES FUYANTES (DATA LEAKAGE)
# =============================================================================

# --- Churn Classification ---
# Ces colonnes sont des proxies DIRECTS ou quasi-directs du Churn.
#
# * Recency (corr=0.86) : Un client parti n'a pas achete depuis longtemps
#   -> fuite parfaite. C'est la principale cause de l'AUC=1.00.
#
# * ChurnRiskCategory (corr=0.88) : Cette colonne est CONSTRUITE pour
#   predire le churn. La mettre en feature revient a donner la reponse.
#
# * CustomerTenureDays (corr=-0.45) : Duree de vie courte = client parti.
#
# * CustomerType_Perdu (corr=0.70) : "Perdu" signifie churne.
#
# * RFMSegment_Dormants (corr=0.58) : "Dormant" = proxy du churn.
#
# * LoyaltyLevel (corr=-0.43), SpendingCategory (corr=-0.38) : Ces
#   categories sont calculees sur l'historique recent -> fuite indirecte.
#
# * AccountStatus_Closed/Suspended : Un compte ferme = parti.
#
# * TenureRatio, MonetaryPerDay : Derives de Recency/Tenure -> fuite.
LEAKY_CHURN_FEATURES = [
    # Proxies temporels directs
    'Recency',
    'CustomerTenureDays',
    'FirstPurchaseDaysAgo',
    'TenureRatio',
    'MonetaryPerDay',
    # Categories construites sur le comportement churn
    'ChurnRiskCategory',
    'LoyaltyLevel',
    'SpendingCategory',
    # One-Hot issues de CustomerType
    'CustomerType_Perdu',
    'CustomerType_Hyperactif',
    'CustomerType_Nouveau',
    'CustomerType_Occasionnel',
    'CustomerType_Regulier',
    # One-Hot issues de RFMSegment
    'RFMSegment_Champions',
    'RFMSegment_Dormants',
    'RFMSegment_Fideles',
    'RFMSegment_Potentiels',
    # One-Hot issues de AccountStatus
    'AccountStatus_Closed',
    'AccountStatus_Suspended',
    'AccountStatus_Active',
    'AccountStatus_Pending',
]

# Variantes avec accents (au cas ou le CSV les contient)
LEAKY_CHURN_FEATURES += [
    'CustomerType_R\u00e9gulier',
    'RFMSegment_Fid\u00e8les',
]

# --- Clustering KMeans ---
# Pour le clustering, on retire Churn (la cible) et toutes les colonnes
# categorielles synthetiques qui resumaient deja le comportement client.
# Raison du Silhouette=0.99 : RFMSegment, CustomerType, SpendingCategory,
# LoyaltyLevel etaient inclus -> le modele clusterisait sur des labels
# pre-existants, pas sur le comportement brut.
# On garde : Frequency, MonetaryTotal, MonetaryAvg, TotalQuantity,
# SatisfactionScore, SupportTicketsCount, Age, etc.
LEAKY_CLUSTERING_FEATURES = [
    'Churn',
    # Categories synthetiques resumant deja le comportement
    'ChurnRiskCategory',
    'SpendingCategory',
    'LoyaltyLevel',
    'BasketSizeCategory',
    # Segments pre-definis = les futurs labels du clustering
    'RFMSegment_Champions',
    'RFMSegment_Dormants',
    'RFMSegment_Fideles',
    'RFMSegment_Potentiels',
    'RFMSegment_Fid\u00e8les',
    'CustomerType_Hyperactif',
    'CustomerType_Nouveau',
    'CustomerType_Occasionnel',
    'CustomerType_Perdu',
    'CustomerType_Regulier',
    'CustomerType_R\u00e9gulier',
    # Statuts de compte biaisent fortement les clusters
    'AccountStatus_Closed',
    'AccountStatus_Suspended',
]