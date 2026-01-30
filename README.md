# Predicting Football Match Outcomes Using Multi-Source Data Fusion

### 1. Project Overview
Solved the challenge of improving short-term match outcome forecasts by fusing heterogeneous data sources, addressing the need for higher predictive accuracy in sports strategy and betting markets.

### 2. Dataset
Integrated data from three primary sources:
* **Kaggle European Soccer Database:** Historical match results and team data.
* **FBref.com:** Detailed player-level performance metrics.
* **OddsPortal:** Historical betting odds for win/draw/loss probabilities.
* **Scope:** ~25,000 matches across 15 seasons and 300+ clubs.

### 3. Approach
* **Models:** Logistic Regression, Random Forest, and **XGBoost**.
* **Tools:** Python (Pandas, Scikit-learn, XGBoost).
* **Techniques:** Data fusion (joining player stats to match IDs), rolling team form averages (last 5 matches), and normalization of betting odds.

### 4. Results
* **Metrics:** The XGBoost model achieved a **ROC-AUC of 0.78** and an F1-score of 0.66.
* **Insights:** Data fusion improved predictive performance by 10% compared to single-source models. Team form and betting odds were identified as the strongest predictors.

### 5. Key Learnings
I learned that data fusion is where the real value lies in sports analytics—raw data from one source isn't enough. I mastered the art of "Entity Resolution"—matching player and team names across three different databases with inconsistent naming conventions—and learned how to engineer time-series features like "Rolling Form" to capture momentum.
