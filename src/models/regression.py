"""
Regression experiments for ResumeMatch AI (Unit 5).
Compares Linear Regression, Multiple Linear Regression, and Polynomial Regression on candidate placement & salary data.
Calculates MAE, MSE, RMSE, and R2 Score.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_salary_regression.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_regression_experiments():
    print("=== Running Unit 5: Supervised Learning - Regression Experiments ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Feature Selection for Multiple Regression:
    # Academic & employability indicators: ssc_p, hsc_p, degree_p, etest_p, mba_p, workex_num
    feature_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "workex_num"]
    target_col = "salary"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Train-test split (80-20 with fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # 1. Simple Linear Regression (Single primary predictor: etest_p / degree_p)
    print("  1. Training Simple Linear Regression (degree_p -> salary)...")
    slr = LinearRegression()
    X_train_slr = X_train[["degree_p"]]
    X_test_slr = X_test[["degree_p"]]
    slr.fit(X_train_slr, y_train)
    y_pred_slr = slr.predict(X_test_slr)
    
    mae_slr = mean_absolute_error(y_test, y_pred_slr)
    mse_slr = mean_squared_error(y_test, y_pred_slr)
    rmse_slr = np.sqrt(mse_slr)
    r2_slr = r2_score(y_test, y_pred_slr)

    results["Simple Linear Regression"] = {
        "MAE": round(float(mae_slr), 2),
        "MSE": round(float(mse_slr), 2),
        "RMSE": round(float(rmse_slr), 2),
        "R2": round(float(r2_slr), 4),
        "features": ["degree_p"]
    }

    # 2. Multiple Linear Regression (All academic + test + experience features)
    print("  2. Training Multiple Linear Regression (all features)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlr = LinearRegression()
    mlr.fit(X_train_scaled, y_train)
    y_pred_mlr = mlr.predict(X_test_scaled)

    mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
    mse_mlr = mean_squared_error(y_test, y_pred_mlr)
    rmse_mlr = np.sqrt(mse_mlr)
    r2_mlr = r2_score(y_test, y_pred_mlr)

    results["Multiple Linear Regression"] = {
        "MAE": round(float(mae_mlr), 2),
        "MSE": round(float(mse_mlr), 2),
        "RMSE": round(float(rmse_mlr), 2),
        "R2": round(float(r2_mlr), 4),
        "features": feature_cols
    }

    # 3. Polynomial Regression (Degree = 2)
    print("  3. Training Polynomial Regression (degree=2)...")
    poly_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    poly_pipeline.fit(X_train[["etest_p", "degree_p"]], y_train)
    y_pred_poly = poly_pipeline.predict(X_test[["etest_p", "degree_p"]])

    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    r2_poly = r2_score(y_test, y_pred_poly)

    results["Polynomial Regression (Degree 2)"] = {
        "MAE": round(float(mae_poly), 2),
        "MSE": round(float(mse_poly), 2),
        "RMSE": round(float(rmse_poly), 2),
        "R2": round(float(r2_poly), 4),
        "features": ["etest_p", "degree_p"]
    }

    # Save models
    joblib.dump({"model": mlr, "scaler": scaler, "features": feature_cols}, MODELS_DIR / "regression_mlr.joblib")
    joblib.dump(poly_pipeline, MODELS_DIR / "regression_poly.joblib")

    # Generate Visualization of Predictions vs Actuals
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_test, color="#1E3A8A", label="Actual Salary", s=50, alpha=0.8)
    plt.plot(range(len(y_pred_mlr)), y_pred_mlr, color="#10B981", linestyle="--", marker="o", label="MLR Predicted Salary", alpha=0.8)
    plt.plot(range(len(y_pred_poly)), y_pred_poly, color="#F59E0B", linestyle=":", marker="s", label="Poly Predicted Salary", alpha=0.8)
    plt.title("Unit 5: Regression Models - Actual vs Predicted Salary", fontsize=13, fontweight="bold")
    plt.xlabel("Test Sample Index", fontsize=11)
    plt.ylabel("Salary (INR)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regression_predictions.png", dpi=150)
    plt.close()

    # Save metrics JSON
    with open(REPORTS_DIR / "regression_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("  [OK] Regression metrics saved:")
    for model_name, metrics in results.items():
        print(f"     {model_name}: MAE={metrics['MAE']:,}, RMSE={metrics['RMSE']:,}, R2={metrics['R2']}")
    return results

if __name__ == "__main__":
    run_regression_experiments()
