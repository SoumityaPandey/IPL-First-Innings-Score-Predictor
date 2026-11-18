import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "IPL.csv"
OUTPUT_DIR = "ipl_project_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. DATASET SELECTION AND HANDLING
raw = pd.read_csv(DATA_PATH, low_memory=False)

print("1. DATASET SELECTION AND HANDLING")
print("Raw Samples:", raw.shape[0])
print("Raw Columns:", raw.shape[1])
print("Unique IPL Matches:", raw["match_id"].nunique())
print("Data Type Summary:\n", raw.dtypes.value_counts())
print("\nFirst Five Records:\n", raw.head())

# One row per match: first innings only
first_innings = raw[raw["innings"] == 1].copy()

rows = []
for match_id, group in first_innings.groupby("match_id", sort=False):
    powerplay = group[group["over"] < 6]
    rows.append({
        "match_id": match_id,
        "date": group["date"].iloc[0],
        "season": group["season"].iloc[0],
        "batting_team": group["batting_team"].iloc[0],
        "bowling_team": group["bowling_team"].iloc[0],
        "venue": group["venue"].iloc[0],
        "powerplay_runs": powerplay["runs_total"].sum(),
        "powerplay_fours": (powerplay["runs_batter"] == 4).sum(),
        "powerplay_sixes": (powerplay["runs_batter"] == 6).sum(),
        "powerplay_dot_balls": ((powerplay["runs_total"] == 0) & (powerplay["valid_ball"] == 1)).sum(),
        "powerplay_wickets": powerplay["wicket_kind"].notna().sum(),
        "powerplay_extras": powerplay["runs_extras"].sum(),
        "final_score": group["runs_total"].sum()
    })

match_df = pd.DataFrame(rows)
features = [
    "powerplay_runs",
    "powerplay_fours",
    "powerplay_sixes",
    "powerplay_dot_balls",
    "powerplay_wickets",
    "powerplay_extras"
]
target = "final_score"

print("\nProcessed Samples:", len(match_df))
print("Features Used:", len(features))
print("Features:", features)
print("\nProcessed Data Types:\n", match_df[features + [target]].dtypes)

# 2. DATA PREPROCESSING
print("\n2. DATA PRE-PROCESSING")
print("Missing Values Before:\n", match_df[features + [target]].isna().sum())

clean = match_df.copy()
for column in features + [target]:
    clean[column] = clean[column].fillna(clean[column].median())

outlier_counts = {}
for column in features:
    q1 = clean[column].quantile(0.25)
    q3 = clean[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_counts[column] = int(((clean[column] < lower) | (clean[column] > upper)).sum())
    clean[column] = clean[column].clip(lower, upper)

print("Missing Values After:\n", clean[features + [target]].isna().sum())
print("Outliers handled using the IQR method.")

# 3. EDA
print("\n3. EXPLORATORY DATA ANALYSIS")
stats = clean[features].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).T
stats["variance"] = clean[features].var()
stats["skewness"] = clean[features].skew()
stats["kurtosis"] = clean[features].kurt()
print("\nDescriptive Statistics:\n", stats.round(3))

# 4. ONE MODEL ONLY: LINEAR REGRESSION
print("\n4. MODEL SELECTION & TRAINING")
X = clean[features]
y = clean[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 5. EVALUATION
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model: Linear Regression")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("\n5. MODEL EVALUATION")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Save outputs
clean["predicted_final_score"] = model.predict(scaler.transform(clean[features]))
clean.to_csv(os.path.join(OUTPUT_DIR, "IPL_ML_Processed.csv"), index=False)
stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))
joblib.dump(model, os.path.join(OUTPUT_DIR, "linear_regression_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "standard_scaler.joblib"))

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as file:
    json.dump({
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "matches": int(raw["match_id"].nunique()),
        "processed_rows": int(clean.shape[0]),
        "features": features,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }, file, indent=2)

# Visualizations
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(clean["final_score"], bins=20, edgecolor="black")
ax.set_title("Distribution of IPL First-Innings Final Scores")
ax.set_xlabel("Final Score")
ax.set_ylabel("Number of Matches")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "histogram.png"), dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(
    [clean["powerplay_runs"], clean["powerplay_fours"], clean["powerplay_sixes"], clean["powerplay_wickets"]],
    tick_labels=["Powerplay Runs", "Fours", "Sixes", "Wickets"]
)
ax.set_title("Boxplot of Selected Powerplay Features")
ax.set_ylabel("Value")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "boxplot.png"), dpi=160)
plt.close(fig)

corr = clean[features + [target]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=65, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Correlation Heatmap")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.65)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
ax.plot([low, high], [low, high], linestyle="--")
ax.set_title("Actual vs Predicted First-Innings Score")
ax.set_xlabel("Actual Final Score")
ax.set_ylabel("Predicted Final Score")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=160)
plt.close(fig)

print("\nAll outputs saved in:", OUTPUT_DIR)
