from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --------------------------
# Load trained models
# --------------------------
reg_model = joblib.load("xgb_reg_log_price.pkl")
clf_model = joblib.load("xgb_clf_price_category.pkl")
cluster_bundle = joblib.load("kmeans_products.pkl")

scaler_cluster = cluster_bundle["scaler"]
imputer_cluster = cluster_bundle.get("imputer", None)
kmeans = cluster_bundle["kmeans"]
cluster_features = cluster_bundle["features"]

# Mapping for categories (must match notebook)
inv_price_mapping = {
    0: "Bas",
    1: "Moyen",
    2: "Élevé"
}

# --------------------------
# Serve HTML UI
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------
# Prediction API endpoint
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # --------------------------
        # Build DataFrame for XGBoost models
        # --------------------------
        x_df = pd.DataFrame([{
            "main_category": data["main_category"],
            "actual_price_clean": float(data["actual_price_clean"]),
            "discount_pct": float(data["discount_pct"]),
            "rating_clean": float(data["rating_clean"]),
            "rating_count_clean": float(data["rating_count_clean"]),
            "popularity_score": float(data["popularity_score"])
        }])

        # --------------------------
        # Price Regression Prediction
        # --------------------------
        pred_log_price = float(reg_model.predict(x_df)[0])
        pred_discounted_price = float(np.exp(pred_log_price))

        # --------------------------
        # Classification Prediction
        # --------------------------
        pred_cat_num = int(clf_model.predict(x_df)[0])
        pred_cat_label = inv_price_mapping.get(pred_cat_num, "Inconnu")

        # --------------------------
        # Clustering Prediction
        # --------------------------
        cluster_input = pd.DataFrame([{
            "actual_price_clean": float(data["actual_price_clean"]),
            "discount_pct": float(data["discount_pct"]),
            "rating_clean": float(data["rating_clean"]),
            "rating_count_clean": float(data["rating_count_clean"]),
            "popularity_score": float(data["popularity_score"]),
            "log_price": pred_log_price   # best approximation
        }])[cluster_features]

        # Impute + scale for clustering
        if imputer_cluster:
            cluster_input = imputer_cluster.transform(cluster_input)

        cluster_scaled = scaler_cluster.transform(cluster_input)
        cluster_idx = int(kmeans.predict(cluster_scaled)[0])

        # Optional: cluster description
        descriptions = {
            0: "Produits entrée de gamme, très accessibles",
            1: "Produits grand public, bon rapport qualité/prix",
            2: "Milieu de gamme",
            3: "Haut de gamme ou premium",
            4: "Catégorie mixte / variable"
        }

        # --------------------------
        # Return all predictions
        
        return jsonify({
            "predicted_discounted_price": pred_discounted_price,
            "price_category_num": pred_cat_num,
            "price_category_label": pred_cat_label,
            "cluster": cluster_idx,
            "cluster_description": descriptions.get(cluster_idx, "N/A")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------
# Run Flask server
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
