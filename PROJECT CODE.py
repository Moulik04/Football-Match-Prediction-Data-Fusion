"""
Title: Predicting Football Match Outcomes Using Multi-Source Data Fusion
Author: Moulik Jain
Contact Member: Moulik Jain
"""

# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
import sqlite3
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2. Convert SQLite to CSV
# =========================
DB_PATH = "database.sqlite"
CSV_DIR = "csv_data"
os.makedirs(CSV_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

for table in tables["name"]:
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df.to_csv(f"{CSV_DIR}/{table}.csv", index=False)

conn.close()
print("SQLite database exported to CSV")

# =========================
# 3. Load Datasets
# =========================

# Match results (Big 5 leagues)
matches = pd.read_csv("BIG_FIVE_1995-2019.csv")

# Kaggle tables
matches_kaggle = pd.read_csv("csv_data/Match.csv")
teams = pd.read_csv("csv_data/Team.csv")
players = pd.read_csv("csv_data/Player.csv")
player_attr = pd.read_csv("csv_data/Player_Attributes.csv")

# Odds / extra stats (archive.zip)
with zipfile.ZipFile("archive.zip", "r") as zip_ref:
    zip_ref.extractall("odds_data")

odds = pd.read_csv("odds_data/odds.csv")

print("All datasets loaded")

# =========================
# 4. Data Cleaning & Fusion
# =========================

# Match outcome
matches_kaggle["home_win"] = (
    matches_kaggle["home_team_goal"] > matches_kaggle["away_team_goal"]
).astype(int)

# Aggregate player ratings per match
player_attr = player_attr.sort_values("date")
player_attr = player_attr.groupby("player_api_id").tail(1)

player_avg_rating = (
    player_attr.groupby("player_api_id")["overall_rating"]
    .mean()
    .reset_index()
)

# Merge teams
matches_kaggle = matches_kaggle.merge(
    teams[["team_api_id", "team_long_name"]],
    left_on="home_team_api_id",
    right_on="team_api_id",
    how="left"
).rename(columns={"team_long_name": "home_team"})

matches_kaggle = matches_kaggle.merge(
    teams[["team_api_id", "team_long_name"]],
    left_on="away_team_api_id",
    right_on="team_api_id",
    how="left"
).rename(columns={"team_long_name": "away_team"})

# Merge odds
data = matches_kaggle.merge(
    odds,
    left_on=["date", "home_team", "away_team"],
    right_on=["date", "HomeTeam", "AwayTeam"],
    how="inner"
)

# =========================
# 5. Feature Engineering
# =========================

# Implied probabilities
data["home_win_prob"] = 1 / data["B365H"]
data["draw_prob"] = 1 / data["B365D"]
data["away_win_prob"] = 1 / data["B365A"]

prob_sum = data[["home_win_prob", "draw_prob", "away_win_prob"]].sum(axis=1)
data[["home_win_prob", "draw_prob", "away_win_prob"]] /= prob_sum.values.reshape(-1, 1)

# Rolling team form
data["home_team_form"] = (
    data.groupby("home_team")["home_team_goal"]
    .rolling(5).mean().reset_index(level=0, drop=True)
)

data["away_team_form"] = (
    data.groupby("away_team")["away_team_goal"]
    .rolling(5).mean().reset_index(level=0, drop=True)
)

data.dropna(inplace=True)

features = [
    "home_team_form",
    "away_team_form",
    "home_win_prob",
    "draw_prob",
    "away_win_prob"
]

X = data[features]
y = data["home_win"]

# =========================
# 6. Exploratory Data Analysis
# =========================
print(X.describe())

plt.figure(figsize=(8,5))
sns.histplot(data["home_team_form"], bins=30, kde=True)
plt.title("Home Team Rolling Form")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=y, y=data["home_win_prob"])
plt.title("Odds vs Outcome")
plt.show()

# =========================
# 7. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 8. Model Training
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss"
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    results.append([
        name,
        accuracy_score(y_test, pred),
        f1_score(y_test, pred),
        roc_auc_score(y_test, prob)
    ])

results_df = pd.DataFrame(
    results, columns=["Model","Accuracy","F1","ROC-AUC"]
)

print(results_df)

# =========================
# 9. Post-Hoc Analysis
# =========================
rf = models["Random Forest"]
importance = rf.feature_importances_

plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

# =========================
# 10. Sensitivity Analysis
# =========================
X_test_mod = X_test.copy()
X_test_mod[:, features.index("home_win_prob")] *= 0.9

delta = np.mean(
    np.abs(
        rf.predict_proba(X_test)[:,1] -
        rf.predict_proba(X_test_mod)[:,1]
    )
)

print("Average probability change after odds perturbation:", round(delta,4))

# =========================
# 11. Generative AI Usage
# =========================
"""
Generative AI was used for:
- Experimental design
- Feature engineering suggestions
- Debugging merge errors
- Model selection guidance
- Post-hoc analysis ideas
"""
