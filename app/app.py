"""
app.py — Application Flask de prédiction client
Projet : Analyse Comportementale Clientèle Retail

Routes :
  GET  /           → Page d'accueil (dashboard)
  GET  /predict    → Formulaire de prédiction
  POST /predict    → Résultat de la prédiction Churn
  GET  /segments   → Visualisation des segments KMeans
  GET  /about      → Description du projet
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# ── Chemins ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'train_test')

app = Flask(__name__)

# ── Chargement des modèles au démarrage ──────────────────────
rf_model  = joblib.load(os.path.join(MODELS_DIR, 'rf_churn_model.pkl'))
km_model  = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
lr_model  = joblib.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
sc_reg    = joblib.load(os.path.join(MODELS_DIR, 'scaler_regression.pkl'))

# Colonnes attendues par le RF (73 features)
X_train_sample = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
LEAKY = [c for c in X_train_sample.columns if any(p in c for p in
    ['Recency','MonetaryPerDay','TenureRatio','ChurnRisk',
     'CustomerType_','RFMSegment_','LoyaltyLevel'])]
RF_FEATURES  = X_train_sample.drop(columns=LEAKY).columns.tolist()
PCA_FEATURES = [f'PC{i+1}' for i in range(10)]

# Noms des segments clustering
CLUSTER_NAMES = {
    0: ('Clients Occasionnels',    'warning',  'fa-clock',         '~15% churn'),
    1: ('Clients VIP / Fidèles',   'success',  'fa-star',          '~4% churn'),
    2: ('Clients à Risque',        'danger',   'fa-exclamation-triangle', '~50% churn'),
    3: ('Clients Inactifs',        'dark',     'fa-moon',          '~100% churn'),
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def build_rf_input(form):
    """
    Construit le vecteur de features pour le Random Forest
    à partir des données du formulaire Flask.
    Les features non saisies sont mises à 0.
    """
    row = {col: 0.0 for col in RF_FEATURES}

    # Features numériques directes
    num_fields = [
        'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
        'MonetaryMin', 'MonetaryMax', 'TotalQuantity',
        'AvgQuantityPerTransaction', 'MinQuantity', 'MaxQuantity',
        'CustomerTenureDays', 'FirstPurchaseDaysAgo',
        'PreferredDayOfWeek', 'PreferredHour', 'PreferredMonth',
        'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases',
        'UniqueProducts', 'UniqueDescriptions', 'AvgProductsPerTransaction',
        'UniqueCountries', 'NegativeQuantityCount', 'ZeroPriceCount',
        'CancelledTransactions', 'ReturnRatio', 'TotalTransactions',
        'UniqueInvoices', 'AvgLinesPerInvoice', 'Age',
        'SupportTicketsCount', 'SatisfactionScore',
        'AgeCategory', 'SpendingCategory', 'PreferredTimeOfDay',
        'BasketSizeCategory', 'RegYear', 'RegMonth', 'RegDay', 'RegWeekday',
        'IP_IsPrivate', 'IP_FirstOctet', 'AvgBasketValue', 'Country_encoded',
    ]
    for f in num_fields:
        if f in RF_FEATURES:
            try:
                row[f] = float(form.get(f, 0) or 0)
            except ValueError:
                row[f] = 0.0

    # One-Hot : FavoriteSeason
    season = form.get('FavoriteSeason', '')
    for s in ['Automne', 'Hiver', 'Printemps', 'Été']:
        key = f'FavoriteSeason_{s}'
        if key in RF_FEATURES:
            row[key] = 1.0 if season == s else 0.0

    # One-Hot : Region
    region = form.get('Region', '')
    for r in ['Afrique','Amérique_du_Nord','Amérique_du_Sud','Asie','Autre',
              'Europe_centrale','Europe_continentale','Europe_de_lEst',
              'Europe_du_Nord','Europe_du_Sud','Moyen-Orient','Océanie','UK']:
        key = f'Region_{r}'
        if key in RF_FEATURES:
            row[key] = 1.0 if region == r else 0.0

    # One-Hot : WeekendPreference
    wp = form.get('WeekendPreference', '')
    for w in ['Inconnu', 'Semaine', 'Weekend']:
        key = f'WeekendPreference_{w}'
        if key in RF_FEATURES:
            row[key] = 1.0 if wp == w else 0.0

    # One-Hot : ProductDiversity
    pd_val = form.get('ProductDiversity', '')
    for p in ['Explorateur', 'Modéré', 'Spécialisé']:
        key = f'ProductDiversity_{p}'
        if key in RF_FEATURES:
            row[key] = 1.0 if pd_val == p else 0.0

    # One-Hot : Gender
    gender = form.get('Gender', '')
    for g in ['F', 'M', 'Unknown']:
        key = f'Gender_{g}'
        if key in RF_FEATURES:
            row[key] = 1.0 if gender == g else 0.0

    # One-Hot : AccountStatus
    status = form.get('AccountStatus', '')
    for s in ['Active', 'Closed', 'Pending', 'Suspended']:
        key = f'AccountStatus_{s}'
        if key in RF_FEATURES:
            row[key] = 1.0 if status == s else 0.0

    return pd.DataFrame([row])[RF_FEATURES]


def predict_churn(df_input):
    """Prédit le Churn et retourne label + probabilité."""
    prob  = rf_model.predict_proba(df_input)[0][1]
    label = int(rf_model.predict(df_input)[0])
    return label, round(float(prob), 4)


def predict_cluster(df_input):
    """Prédit le cluster KMeans (sur les 10 premières PCA)."""
    # On utilise les features disponibles pour approximer les composantes PCA
    X_pca_sample = pd.read_csv(os.path.join(DATA_DIR, 'X_train_pca.csv')).iloc[:10]
    # Utiliser le centroïde le plus proche basé sur les features numériques disponibles
    X_train_full = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    LEAKY2 = [c for c in X_train_full.columns if any(p in c for p in
        ['Recency','MonetaryPerDay','TenureRatio','ChurnRisk',
         'CustomerType_','RFMSegment_','LoyaltyLevel'])]
    common_cols = [c for c in RF_FEATURES if c in X_train_full.columns]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X_pca_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_pca.csv')).iloc[:, :10]

    # Simple: standardize input using training set stats and assign cluster
    X_tr = X_train_full[common_cols].fillna(0)
    sc   = StandardScaler()
    sc.fit(X_tr)
    X_in = df_input[common_cols].fillna(0)
    X_in_sc = sc.transform(X_in)

    pca = PCA(n_components=10, random_state=42)
    pca.fit(sc.transform(X_tr))
    X_in_pca = pca.transform(X_in_sc)

    cluster = int(km_model.predict(X_in_pca)[0])
    return cluster


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Page d'accueil avec métriques du projet."""
    stats = {
        'n_clients':  4372,
        'n_features': 87,
        'churn_rate': '33.3%',
        'rf_auc':     '0.99',
        'rf_f1':      '0.95',
        'reg_r2':     '0.77',
        'n_clusters': 4,
    }
    return render_template('index.html', stats=stats)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Formulaire + résultat de prédiction."""
    result = None

    if request.method == 'POST':
        form = request.form

        # Construire le vecteur d'entrée
        df_input = build_rf_input(form)

        # Prédictions
        churn_label, churn_prob = predict_churn(df_input)
        cluster_id = predict_cluster(df_input)

        cluster_name, cluster_color, cluster_icon, cluster_churn = \
            CLUSTER_NAMES.get(cluster_id, ('Inconnu', 'secondary', 'fa-question', '-'))

        # Niveau de risque
        if churn_prob >= 0.75:
            risk_level, risk_color = 'Critique', 'danger'
        elif churn_prob >= 0.5:
            risk_level, risk_color = 'Élevé', 'warning'
        elif churn_prob >= 0.25:
            risk_level, risk_color = 'Modéré', 'info'
        else:
            risk_level, risk_color = 'Faible', 'success'

        result = {
            'churn_label':    churn_label,
            'churn_prob':     round(churn_prob * 100, 1),
            'churn_text':     'Client Parti (Churned)' if churn_label == 1 else 'Client Fidèle',
            'churn_color':    'danger' if churn_label == 1 else 'success',
            'risk_level':     risk_level,
            'risk_color':     risk_color,
            'cluster_id':     cluster_id,
            'cluster_name':   cluster_name,
            'cluster_color':  cluster_color,
            'cluster_icon':   cluster_icon,
            'cluster_churn':  cluster_churn,
            'form_data':      dict(form),
        }

    return render_template('predict.html', result=result)


@app.route('/segments')
def segments():
    """Page de visualisation des 4 segments clients."""
    segments_data = [
        {
            'id':          1,
            'name':        'Clients VIP / Fidèles',
            'color':       'success',
            'icon':        'fa-star',
            'size':        '30%',
            'churn':       '~4%',
            'frequency':   'Élevée (> 5 cmds)',
            'monetary':    'Élevé (> 1 500 £)',
            'tenure':      'Longue (> 200 jours)',
            'satisfaction':'Bonne',
            'action':      'Programme fidélité premium, ventes privées, offres exclusives.',
            'badge':       'Priorité : Fidéliser',
        },
        {
            'id':          0,
            'name':        'Clients Occasionnels',
            'color':       'warning',
            'icon':        'fa-clock',
            'size':        '45%',
            'churn':       '~15%',
            'frequency':   'Faible (1-3 cmds)',
            'monetary':    'Moyen',
            'tenure':      'Courte',
            'satisfaction':'Moyenne',
            'action':      'Promotions saisonnières, emails de réactivation, cross-selling.',
            'badge':       'Priorité : Engager',
        },
        {
            'id':          2,
            'name':        'Clients à Risque',
            'color':       'danger',
            'icon':        'fa-exclamation-triangle',
            'size':        '0.1%',
            'churn':       '~50%',
            'frequency':   'Faible',
            'monetary':    'Négatif (retours)',
            'tenure':      'Variable',
            'satisfaction':'Faible',
            'action':      'Enquête satisfaction urgente, traitement des retours, geste commercial.',
            'badge':       'Priorité : Retenir',
        },
        {
            'id':          3,
            'name':        'Clients Inactifs / Perdus',
            'color':       'dark',
            'icon':        'fa-moon',
            'size':        '25%',
            'churn':       '~100%',
            'frequency':   'Très faible',
            'monetary':    'Très faible',
            'tenure':      'Très courte',
            'satisfaction':'Inconnue',
            'action':      'Campagne de réactivation finale ou suppression de la base client.',
            'badge':       'Priorité : Réactiver',
        },
    ]
    return render_template('segments.html', segments=segments_data)


@app.route('/about')
def about():
    """Page de description du projet."""
    pipeline = [
        ('1', 'Exploration',     'Analyse qualité, distributions, corrélations',        'fa-search',    'primary'),
        ('2', 'Préparation',     'Nettoyage, encodage, feature engineering',             'fa-tools',     'info'),
        ('3', 'Transformation',  'Réduction de dimension par ACP (87 → 44 composantes)','fa-compress',  'warning'),
        ('4', 'Modélisation',    'KMeans, Random Forest, Régression Linéaire',           'fa-brain',     'success'),
        ('5', 'Évaluation',      'Métriques, courbes ROC, recommandations métier',       'fa-chart-bar', 'danger'),
        ('6', 'Déploiement',     'Interface Flask avec prédiction en temps réel',        'fa-rocket',    'dark'),
    ]
    models_info = [
        ('KMeans (k=4)',         'Clustering',           'Silhouette = 0.21',  'Segmentation clients'),
        ('Random Forest',        'Classification Churn', 'AUC = 0.99 / F1 = 0.95', 'Prédiction churn'),
        ('Régression Linéaire',  'Régression',           'R² = 0.77 / MAE = 789£', 'Prédiction CA'),
    ]
    return render_template('about.html', pipeline=pipeline, models_info=models_info)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API JSON pour la prédiction.
    Entrée  : JSON avec les features du client
    Sortie  : { churn, probability, cluster }
    """
    data = request.get_json(force=True)
    df_input = build_rf_input(data)
    churn_label, churn_prob = predict_churn(df_input)
    cluster_id = predict_cluster(df_input)
    cluster_name = CLUSTER_NAMES.get(cluster_id, ('Inconnu',))[0]

    return jsonify({
        'churn':        churn_label,
        'probability':  churn_prob,
        'risk':         'Élevé' if churn_prob >= 0.5 else 'Faible',
        'cluster':      cluster_id,
        'cluster_name': cluster_name,
    })


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Démarrage de l'application Flask")
    print("  URL : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
