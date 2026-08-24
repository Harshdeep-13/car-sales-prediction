"""
train_model.py
---------------
Trains a car resale price prediction model on cleaned_car_data.csv
and saves the trained pipeline (preprocessing + model) to model.pkl.

Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
DATA_PATH = "cleaned_car_data.csv"   # place this file next to the script
df = pd.read_csv(DATA_PATH)

print("Rows, cols:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------
CURRENT_YEAR = datetime.now().year
df["Car_Age"] = CURRENT_YEAR - df["Year"]
df.drop(columns=["Year"], inplace=True)

# Car_Name has 98 unique values -> too many for one-hot encoding cleanly.
# We keep it out of the model but you could target-encode it later if needed.
df.drop(columns=["Car_Name"], inplace=True)

TARGET = "Selling_Price"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]

print("\nFeatures used:", list(X.columns))

# ---------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 4. Preprocessing + model pipeline
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# ---------------------------------------------------------------
# 5. Hyperparameter tuning (small, fast search)
# ---------------------------------------------------------------
param_dist = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [None, 5, 8, 10, 15],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)
rf_best = search.best_estimator_
print("\nBest RF params:", search.best_params_)

# ---------------------------------------------------------------
# 6. Evaluate multiple candidates and keep the best one
# ---------------------------------------------------------------
def evaluate(model, name):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\n{name}")
    print(f"  R2  : {r2:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}

candidates = {}

lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
lr_pipeline.fit(X_train, y_train)
candidates["LinearRegression"] = (lr_pipeline, evaluate(lr_pipeline, "Linear Regression (baseline)"))

candidates["RandomForest (tuned)"] = (rf_best, evaluate(rf_best, "Random Forest (tuned)"))

gbr_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", GradientBoostingRegressor(random_state=42))])
gbr_pipeline.fit(X_train, y_train)
candidates["GradientBoosting"] = (gbr_pipeline, evaluate(gbr_pipeline, "Gradient Boosting"))

# Pick the model with the highest R2 on the held-out test set
best_name = max(candidates, key=lambda k: candidates[k][1]["r2"])
best_model, best_metrics = candidates[best_name]
print(f"\n>>> Selected best model: {best_name} (R2={best_metrics['r2']:.4f})")

# ---------------------------------------------------------------
# 7. Save the winning model + metadata
# ---------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

metadata = {
    "trained_on": datetime.now().isoformat(),
    "features": list(X.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categorical_values": {c: sorted(X[c].unique().tolist()) for c in categorical_features},
    "target": TARGET,
    "best_model_name": best_name,
    "metrics": best_metrics,
    "current_year_used_for_training": CURRENT_YEAR,
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model.pkl and model_metadata.json")
