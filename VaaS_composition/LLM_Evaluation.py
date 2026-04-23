#################### COMPUTE METRICS for CLASSIFICATION EVALUATION # VERSION 4 --Valid and raunable########################

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error
)

# --- Load CSV files ---
pred_df = pd.read_csv("Zero_Shot_classified_llmgptCoT.csv")
truth_df = pd.read_csv("Zero_Shot_Ground_Truth.csv")

# --- Ensure same order using Request as key (IMPORTANT) ---
df = pd.merge(truth_df, pred_df, on="Request", suffixes=("_true", "_pred"))

# ================================
# 1. CLASSIFICATION EVALUATION
# ================================
y_true = df["Class_true"]
y_pred = df["Class_pred"]

print("=== Classification Evaluation ===")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.2%}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"MCC      : {matthews_corrcoef(y_true, y_pred):.4f}\n")

# ================================
# 2. OPTIMIZATION EVALUATION
# ================================

# Convert to numeric (handle empty cells)
df["Best_Score_true"] = pd.to_numeric(df["Best_Score_true"], errors="coerce")
df["Best_Score_pred"] = pd.to_numeric(df["Best_Score_pred"], errors="coerce")

# Keep only valid pairs
opt_df = df.dropna(subset=["Best_Score_true", "Best_Score_pred"])

if len(opt_df) == 0:
    print("No valid optimization data found.")
else:
    y_true_score = opt_df["Best_Score_true"]
    y_pred_score = opt_df["Best_Score_pred"]

    mae = mean_absolute_error(y_true_score, y_pred_score)
    mse = mean_squared_error(y_true_score, y_pred_score)
    rmse = np.sqrt(mse)

    # Relative Error (AROL)
    relative_errors = np.abs(y_pred_score - y_true_score) / np.abs(y_true_score)
    arol = np.mean(relative_errors)

    print("=== Optimization Evaluation ===")
    print(f"Matched Samples: {len(opt_df)}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"AROL : {arol:.4f}")

# ================================
# 3. DEBUG / ANALYSIS TABLE
# ================================

comparison_df = opt_df.copy()

comparison_df["Absolute_Error"] = np.abs(
    comparison_df["Best_Score_pred"] - comparison_df["Best_Score_true"]
)

comparison_df["Relative_Error"] = (
    comparison_df["Absolute_Error"] / np.abs(comparison_df["Best_Score_true"])
)

print("\n=== Sample Comparison ===")
print(comparison_df.head(20))

print("\n=== Worst Cases ===")
print(comparison_df.sort_values(by="Absolute_Error", ascending=False).head(10))

print("\n=== Best Cases ===")
print(comparison_df.sort_values(by="Absolute_Error", ascending=True).head(10))

# Save for paper/debug
comparison_df.to_csv("comparison_debug.csv", index=False)

print("\n✔ File saved: comparison_debug.csv")