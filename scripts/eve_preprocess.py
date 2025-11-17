#!/usr/bin/env python3

"""
Script complet pour l'analyse des logs Suricata (eve.json) d'un honeypot T-Pot.

Ce script effectue :
1.  Chargement et Aplatissement (Flattening) des logs JSON-lines.
2.  Sélection d'un sous-ensemble de features pertinentes.
3.  Détection dynamique des types de features (numérique, catégorielle).
4.  Encodage des features catégorielles basé sur la cardinalité :
    - Faible cardinalité (< 50) : One-Hot Encoding
    - Haute cardinalité (>= 50) : Ordinal Encoding
5.  Visualisation des outliers (Boxplots) et de la cardinalité.
6.  Création d'un pipeline de prétraitement robuste (RobustScaler, Imputers).
7.  Réduction de dimension (PCA).
8.  Entraînement d'un modèle de détection d'anomalies (Isolation Forest).
9.  Visualisation des résultats (Scree plot PCA, Scatter plot des anomalies).
"""

import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# --- Constantes de Configuration ---
FILE_NAME = 'eve.json.22'  # Le nom de votre fichier de log
CARDINALITY_THRESHOLD = 50       # Seuil pour OHE vs Ordinal
PCA_VARIANCE_RATIO = 0.95      # % de variance à conserver pour le PCA

# Liste des features potentiellement intéressantes à extraire.
# L'aplatissement crée > 500 colonnes, nous devons présélectionner.
POTENTIALLY_USEFUL_FEATURES = [
    'event_type',
    'proto',
    'src_ip',
    'dest_ip',
    'src_port',
    'dest_port',
    'alert.signature',
    'alert.category',
    'alert.severity',
    'flow.bytes_toserver',
    'flow.bytes_toclient',
    'flow.pkts_toserver',
    'flow.pkts_toclient',
    'http.hostname',
    'http.http_method',
    'http.status',
    'dns.query.rrname',
    'dns.query.rrtype',
    'ssh.client.software_version',
    'ssh.server.software_version'
]

# --- 1. Chargement et Aplatissement ---

def load_and_flatten(filepath):
    """Charge un fichier JSON-lines et l'aplatit dans un DataFrame."""
    print(f"Chargement et aplatissement du fichier : {filepath}...")
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignorer les lignes JSON mal formées
                pass
    
    if not data:
        print("Aucune donnée n'a été chargée. Vérifiez le fichier.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes (avant filtrage).")
    
    # Filtrer pour ne garder que les colonnes utiles qui existent
    existing_features = [col for col in POTENTIALLY_USEFUL_FEATURES if col in df.columns]
    X = df[existing_features].copy()
    print(f"DataFrame réduit à {X.shape[1]} features sélectionnées.")
    return X

# --- 2. Détection Dynamique et Pipeline ---

  
    
    # Détecter les types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # Séparer les catégorielles par cardinalité
    low_cardinality_features = []
    high_cardinality_features = []
    
    print("Analyse de la cardinalité...")
    for col in categorical_features:
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= threshold:
            low_cardinality_features.append(col)
        else:
            high_cardinality_features.append(col)

    print(f"  > Numériques ({len(numeric_features)}): {numeric_features}")
    print(f"  > Caté. Faible Card. ({len(low_cardinality_features)}): {low_cardinality_features}")
    print(f"  > Caté. Haute Card. ({len(high_cardinality_features)}): {high_cardinality_features}")

    # Créer les pipelines de transformation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Robuste aux outliers
    ])

    low_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Assembler le préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_low', low_card_transformer, low_cardinality_features),
            ('cat_high', high_card_transformer, high_cardinality_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_features, high_cardinality_features

# --- 3. Visualisations ---

def plot_initial_visuals(X, numeric_features, high_card_features):
    """Affiche les graphiques d'exploration initiaux."""
    print("Génération des graphiques d'exploration...")
    
    # 3.1 Boxplots pour les outliers (limité aux 5 premières pour la lisibilité)
    plt.figure(figsize=(15, 6))
    plt.suptitle('Détection des Outliers (sur données brutes)', fontsize=16)
    
    plot_limit = min(len(numeric_features), 5)
    for i, col in enumerate(numeric_features[:plot_limit]):
        plt.subplot(1, plot_limit, i + 1)
        sns.boxplot(y=X[col])
        plt.title(col)
        plt.yscale('log') # Log-scale est souvent nécessaire pour les données réseau
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 3.2 Barplot pour la cardinalité
    if high_card_features:
        plt.figure(figsize=(12, 6))
        cardinality = X[high_card_features].nunique().sort_values(ascending=False)
        sns.barplot(x=cardinality.values, y=cardinality.index)
        plt.title('Cardinalité des Features "Hautes" (Justifiant OrdinalEncoder)')
        plt.xlabel('Nombre de valeurs uniques')
        plt.tight_layout()

def plot_results(pca, X_pca, anomaly_score):
    """Affiche les graphiques des résultats (PCA et Anomalies)."""
    print("Génération des graphiques de résultats...")

    # 3.3 Scree Plot PCA
    plt.figure(figsize=(10, 5))
    explained_variance = pca.explained_variance_ratio_
    plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
    plt.axhline(y=PCA_VARIANCE_RATIO, color='r', linestyle=':', 
                label=f'{PCA_VARIANCE_RATIO*100}% Variance')
    plt.title('Variance Expliquée Cumulée par les Composantes PCA')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance cumulée')
    plt.legend()
    plt.grid(True)
    
    # 3.4 Scatter Plot des Anomalies
    plt.figure(figsize=(12, 8))
    
    # Créer un DataFrame pour le plotting
    df_plot = pd.DataFrame(X_pca[:, :2], columns=['PCA1', 'PCA2'])
    df_plot['anomaly_score'] = anomaly_score
    
    # Trier pour que les anomalies (scores bas) soient dessinées par-dessus
    df_plot = df_plot.sort_values(by='anomaly_score') 

    sc = sns.scatterplot(
        data=df_plot,
        x='PCA1',
        y='PCA2',
        hue='anomaly_score',
        palette='coolwarm_r', # Les anomalies seront en rouge
        s=10,
        alpha=0.7
    )
    plt.title('Détection d\'Anomalies (Isolation Forest) sur les 2 premières Comp. PCA')
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    
    # Mettre à jour la légende pour la rendre plus lisible
    norm = plt.Normalize(df_plot['anomaly_score'].min(), df_plot['anomaly_score'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=sc.axes)
    cbar.set_label('Score d\'Anomalie (plus bas = plus anormal)')


# --- 4. Exécution Principale ---

def main():
    try:
        X = load_and_flatten(FILE_NAME)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{FILE_NAME}' n'a pas été trouvé.")
        print("Veuillez vérifier le nom et le chemin du fichier.")
        return
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement : {e}")
        return

    if X.empty:
        print("Le DataFrame est vide. Arrêt du script.")
        return
    
    # Créer le préprocesseur
    preprocessor, num_features, high_card_features = create_preprocessor(
        X, CARDINALITY_THRESHOLD
    )
    
    # Afficher les graphiques d'exploration
    plot_initial_visuals(X, num_features, high_card_features)
    
    # --- Création et exécution du pipeline ---
    
    # Nous séparons les étapes pour pouvoir accéder aux objets (pca, model)
    # et pour utiliser les données transformées (X_pca) pour le plotting.

    # 1. Pipeline de Prétraitement
    preprocess_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    print("\nÉtape 1/4 : Prétraitement des données...")
    X_processed = preprocess_pipeline.fit_transform(X)
    print(f"Données prétraitées. Nouvelle forme : {X_processed.shape}")
    
    # 2. Étape PCA
    print("Étape 2/4 : Réduction de dimension (PCA)...")
    pca = PCA(n_components=PCA_VARIANCE_RATIO,svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_processed)
    print(f"Données réduites à {X_pca.shape[1]} composantes (capturant {PCA_VARIANCE_RATIO*100}% de variance).")

    # 3. Étape Modèle
    print("Étape 3/4 : Entraînement du modèle (Isolation Forest)...")
    model = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    model.fit(X_pca)
    
    # 4. Obtenir les scores
    print("Étape 4/4 : Calcul des scores d'anomalie...")
    anomaly_score = model.decision_function(X_pca)
    predictions = model.predict(X_pca) # -1 pour anomalie, 1 pour normal

    # --- Analyse des Résultats ---
    
    # Ajouter les scores au DataFrame original pour analyse
    X['anomaly_score'] = anomaly_score
    X['is_anomaly'] = predictions

    print("\n--- Résultats de la Détection d'Anomalies ---")
    print(f"Nombre d'anomalies détectées (score -1) : {(predictions == -1).sum()}")
    print(f"Nombre d'événements normaux (score 1)   : {(predictions == 1).sum()}")

    print("\n--- Top 15 des événements les plus anormaux ---")
    top_anomalies = X.sort_values(by='anomaly_score').head(15)
    
    # Afficher les colonnes les plus pertinentes pour les anomalies
    cols_to_show = ['anomaly_score', 'event_type', 'src_ip', 'dest_port', 'alert.signature']
    cols_to_show = [col for col in cols_to_show if col in top_anomalies.columns]
    print(top_anomalies[cols_to_show])
    
    # Afficher les graphiques de résultats
    plot_results(pca, X_pca, anomaly_score)
    
    print("\nScript terminé. Affichage des graphiques...")
    plt.show()


if __name__ == "__main__":
    main()