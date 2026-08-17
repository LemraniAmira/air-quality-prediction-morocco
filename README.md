# 🌫️ Prédiction de la qualité de l'air dans les villes marocaines

Projet de Machine Learning appliqué à la prédiction du **PM2.5** (particules fines) dans quatre villes du Maroc — **Casablanca, Rabat, Marrakech et Fès** — comparant une approche **centralisée classique** à une approche **Federated Learning (apprentissage fédéré)**, avec une implémentation manuelle du FedAvg puis une implémentation avec le framework **Flower**.

## 📌 Contexte

Chaque ville dispose de ses propres capteurs/données de pollution et de météo. Plutôt que de centraliser toutes les données sur un seul serveur (ce qui pose des questions de confidentialité et de coût de transfert), le Federated Learning permet d'entraîner un modèle global **sans jamais faire sortir les données locales de chaque ville** — seuls les paramètres du modèle sont échangés et agrégés.

Ce projet compare les deux approches pour évaluer le compromis entre confidentialité (FL) et performance brute (modèle centralisé).

## 🗂️ Structure du dépôt

```
.
├── 01_collecte.ipynb            # Collecte des données via l'API OpenWeatherMap (météo + qualité de l'air)
├── 02_preparation.ipynb         # Nettoyage, feature engineering, encodage, normalisation
├── 03_modeles.ipynb             # Modèles centralisés : Régression Linéaire vs Random Forest
├── 04_federe.ipynb              # Federated Learning — implémentation manuelle du FedAvg
├── 04_federe_flower.ipynb       # Federated Learning avec le framework Flower (NumPyClient, FedAvg)
├── Data/
│   ├── data_raw.csv             # Données brutes collectées (météo + pollution, 4 villes)
│   └── data_clean.csv           # Données nettoyées, encodées et normalisées, prêtes pour le ML
├── Image/                       # Graphiques générés (comparaisons, convergence, importance des features...)
├── Rapport_Projet.pdf           # Rapport complet du projet
└── requirements.txt             # Dépendances Python
```

## 🔬 Pipeline du projet

### Étape 1 — Collecte des données (`01_collecte.ipynb`)
Connexion à l'API **OpenWeatherMap** pour récupérer, pour les 4 villes :
- Données météo : température, humidité, pression, vent, visibilité, nébulosité
- Données de qualité de l'air : AQI, PM2.5, PM10, NO2, O3, CO, SO2

### Étape 2 — Préparation des données (`02_preparation.ipynb`)
- Nettoyage (doublons, valeurs manquantes)
- Extraction de features temporelles à partir du timestamp (heure, jour de la semaine, jour du mois)
- Encodage des variables catégorielles (ville, description météo)
- Normalisation des features numériques
- Sortie : `Data/data_clean.csv` (212 lignes, 20 features, cible = `pm2_5`)

### Étape 3 — Modèles centralisés (`03_modeles.ipynb`)
Split train/test 80/20 (169 / 43 lignes), entraînement et comparaison de deux modèles :

| Modèle | MAE (µg/m³) | RMSE (µg/m³) | R² |
|---|---|---|---|
| Régression Linéaire | 0.4935 | 0.6188 | 0.8140 |
| **Random Forest** | **0.3293** | **0.4630** | **0.8959** |

➡️ **Random Forest** est le meilleur modèle centralisé (89.6 % de variance expliquée). La feature la plus importante est de loin le PM10 (poids ≈ 0.64).

### Étape 4 — Federated Learning manuel (`04_federe.ipynb`)
Chaque ville = un client fédéré (données jamais partagées avec le serveur). Simulation de **10 rounds de FedAvg** avec un `SGDRegressor` comme modèle local :

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Centralisé (référence, LR) | 0.4935 | 0.6188 | 0.8140 |
| Fédéré (FedAvg, manuel) | 0.7545 | 0.9627 | 0.5499 |

➡️ Le modèle fédéré converge (R² passe de -1.52 au round 1 à +0.55 au round 10) mais reste en retrait par rapport au centralisé — compromis attendu en échange de la confidentialité des données.

### Étape 4 bis — Federated Learning avec Flower (`04_federe_flower.ipynb`)
Même principe mais implémenté avec le framework **Flower 1.31** (`NumPyClient`, `FedAvg`, `ndarrays_to_parameters`/`parameters_to_ndarrays`), pour un protocole FL réaliste et standard.

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Centralisé (LR) | 0.4935 | 0.6188 | 0.8140 |
| Flower FedAvg (FL) | 0.8397 | 1.0667 | 0.4474 |

## 🛠️ Stack technique

- Python, pandas, numpy
- scikit-learn (Régression Linéaire, Random Forest, SGDRegressor)
- Flower (`flwr[simulation]`) pour le Federated Learning
- matplotlib, seaborn pour la visualisation
- API OpenWeatherMap pour la collecte de données

## 🚀 Installation

```bash
git clone <url-du-repo>
cd <nom-du-repo>
pip install -r requirements.txt
```

Puis exécuter les notebooks dans l'ordre : `01_collecte.ipynb` → `02_preparation.ipynb` → `03_modeles.ipynb` → `04_federe.ipynb` → `04_federe_flower.ipynb`.

> Pour relancer la collecte de données, une clé API OpenWeatherMap est nécessaire (à configurer dans `01_collecte.ipynb`). Sinon, le dataset déjà collecté est disponible dans `Data/`.

## 📈 Résultats clés

- Le **Random Forest centralisé** est le modèle le plus performant (R² = 0.896).
- Le **Federated Learning** (manuel ou Flower) reste moins précis que le centralisé (écart de R² d'environ 0.26 à 0.37), un compromis cohérent avec le principe de préservation de la confidentialité des données locales.
- Plus de rounds de communication ou plus de données locales par ville amélioreraient la convergence du modèle fédéré.
