#  MLprediction — Prédiction intelligente de prix produits

Modèle de Machine Learning qui prédit le **prix optimal d'un produit** en 
fonction de plusieurs facteurs : le temps (saisonnalité), l'environnement, 
les conditions de marché, et la catégorie du produit.

##  Objectif

Aider à fixer un prix adapté et compétitif en tenant compte de facteurs 
dynamiques plutôt qu'un prix fixe, en s'appuyant sur des données historiques 
(ex. Amazon).

##  Approche

- **Clustering** (K-Means) pour regrouper les produits par profil
- **Classification** (XGBoost) pour prédire une catégorie de prix
- **Régression** (XGBoost) pour prédire le prix précis (log-transformé)

##  Technologies

- Python
- XGBoost, scikit-learn
- Pandas / NumPy
- Flask (`app.py`) pour servir le modèle


