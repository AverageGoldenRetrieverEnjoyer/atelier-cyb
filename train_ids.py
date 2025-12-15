import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Charger les données
print("[*] Chargement du dataset ...")
df = pd.read_csv("dataset_flux.csv")  # Mets ici le nom exact de ton fichier CSV

# 2. Nettoyer les colonnes inutiles (adapte si nécessaire)
cols_to_drop = ['Source IP', 'Destination IP', 'Flow ID', 'Timestamp']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop, errors='ignore')

# 3. Nettoyage des valeurs manquantes
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# 4. Encodage des catégories et de la cible
X = df.drop(columns=['Label'])     # 'Label' = colonne cible, adapte le nom si besoin
y = df['Label']

X = pd.get_dummies(X)              # Encodage One-Hot
le = LabelEncoder()                
y = le.fit_transform(y)            # Ex: "Normal, DDoS" -> 0, 1

# 5. Train/Test split et normalisation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Entraînement du modèle
print("[*] Entraînement du modèle Random Forest ...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1 
)
model.fit(X_train, y_train)

# 7. Évaluation
y_pred = model.predict(X_test)
print("\n--- Rapport Classification ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n--- Matrice de confusion ---")
print(confusion_matrix(y_test, y_pred))

# 8. Facultatif : Afficher l'importance des variables
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

print("\nTop 5 features les plus importantes :")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
