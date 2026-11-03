"""Models and scores. Every model is fitted on earlier seasons and scored on later ones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES = ["H", "D", "A"]  # ordered, which the ranked probability score needs


def one_hot(result: pd.Series) -> np.ndarray:
    return np.column_stack([(result == o).to_numpy(dtype=float) for o in OUTCOMES])


def scores(probs: np.ndarray, result: pd.Series) -> dict[str, float]:
    """Log loss, ranked probability score, Brier score, accuracy and home-win ROC-AUC (lower is better for the first three)."""
    probs = np.clip(probs / probs.sum(axis=1, keepdims=True), 1e-12, 1.0)
    y = one_hot(result)
    cum_p, cum_y = np.cumsum(probs, axis=1)[:, :2], np.cumsum(y, axis=1)[:, :2]
    return {
        "log_loss": float(-np.mean(np.log(np.sum(probs * y, axis=1)))),
        "rps": float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / 2)),
        "brier": float(np.mean(np.sum((probs - y) ** 2, axis=1))),
        "accuracy": float(np.mean(probs.argmax(axis=1) == y.argmax(axis=1))),
        "home_win_auc": float(roc_auc_score(y[:, 0], probs[:, 0])),
    }


def paired_bootstrap(probs_a: np.ndarray, probs_b: np.ndarray, result: pd.Series,
                     n: int = 5000, seed: int = 0) -> dict[str, float]:
    """Mean per-match log-loss difference (a minus b) with a 95% bootstrap interval over matches."""
    y = one_hot(result)

    def per_match(p: np.ndarray) -> np.ndarray:
        p = np.clip(p / p.sum(axis=1, keepdims=True), 1e-12, 1.0)
        return -np.log(np.sum(p * y, axis=1))

    diff = per_match(probs_a) - per_match(probs_b)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "low": float(low), "high": float(high)}


def base_rates(train_result: pd.Series, n: int) -> np.ndarray:
    freq = np.array([np.mean(train_result == o) for o in OUTCOMES])
    return np.tile(freq, (n, 1))


def logistic() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def boosted() -> object:
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, random_state=0)


def fit_predict(model, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model.fit(train[features].to_numpy(), train["result"].to_numpy())
    proba = model.predict_proba(test[features].to_numpy())
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return proba[:, order]


def chronological_split(df: pd.DataFrame, train_seasons: range, test_seasons: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(test_seasons)]
    assert train["date"].max() < test["date"].min(), "training data must end before the test period starts"
    return train, test


def table(rows: list[dict], digits: int = 4) -> str:
    """A markdown table of model scores."""
    frame = pd.DataFrame(rows)
    head = "| " + " | ".join(frame.columns) + " |\n| " + " | ".join("---" for _ in frame.columns) + " |\n"
    body = ""
    for _, row in frame.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body
