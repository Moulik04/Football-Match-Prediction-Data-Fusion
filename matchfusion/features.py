"""Pre-match features. Every value for a match uses only what was known before kick-off.

The original version of this project computed form as a rolling mean that
included the match being predicted. Each function here says how it avoids that,
and tests/test_leakage.py checks it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BOOKMAKERS


def _goal_multiplier(goal_diff: int) -> float:
    """Bigger wins move ratings more (the World Football Elo convention)."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def elo_ratings(matches: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0, start: float = 1500.0) -> pd.DataFrame:
    """Elo ratings computed from the results alone, one league at a time.

    Returns elo_home and elo_away aligned with `matches`: each is the rating
    BEFORE that match, so a match's own result never reaches its features.
    A club joining a league season takes the average final rating of the clubs
    that left that league the season before (usually the relegated clubs).
    """
    elo_home = pd.Series(np.nan, index=matches.index)
    elo_away = pd.Series(np.nan, index=matches.index)
    for _, league in matches.groupby("league", sort=False):
        ratings: dict[str, float] = {}
        previous_teams: set[str] = set()
        for _, season in league.sort_values("date", kind="stable").groupby("season", sort=True):
            teams = set(season["home"]) | set(season["away"])
            if previous_teams:
                departed = previous_teams - teams
                entry = float(np.mean([ratings[t] for t in departed])) if departed else start
                for team in teams - previous_teams:
                    ratings[team] = entry
            for team in teams:
                ratings.setdefault(team, start)
            pre_home, pre_away = [], []
            for home, away, hg, ag in zip(season["home"], season["away"], season["home_goals"], season["away_goals"]):
                rh, ra = ratings[home], ratings[away]
                pre_home.append(rh)
                pre_away.append(ra)
                expected = 1.0 / (1.0 + 10 ** ((ra - rh - home_adv) / 400.0))
                actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
                delta = k * _goal_multiplier(abs(hg - ag)) * (actual - expected)
                ratings[home], ratings[away] = rh + delta, ra - delta
            elo_home.loc[season.index] = pre_home
            elo_away.loc[season.index] = pre_away
            previous_teams = teams
    return pd.DataFrame({"elo_home": elo_home, "elo_away": elo_away})


def rolling_form(matches: pd.DataFrame, window: int = 5, max_gap_days: int = 150) -> pd.DataFrame:
    """Points and goal difference per game over each club's previous `window` matches.

    shift(1) drops the current match before the rolling mean is taken. Form is
    left empty (and later filled with 0, "no information") when a club has
    fewer than 3 earlier matches or its last match was more than
    `max_gap_days` ago, which is how a club promoted after years away shows up.
    """
    home = pd.DataFrame({"match": matches.index, "league": matches["league"], "date": matches["date"],
                         "team": matches["home"], "gf": matches["home_goals"], "ga": matches["away_goals"]})
    away = pd.DataFrame({"match": matches.index, "league": matches["league"], "date": matches["date"],
                         "team": matches["away"], "gf": matches["away_goals"], "ga": matches["home_goals"]})
    long = pd.concat([home.assign(side="home"), away.assign(side="away")], ignore_index=True)
    long["pts"] = np.where(long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0))
    long["gd"] = long["gf"] - long["ga"]
    long = long.sort_values(["league", "team", "date"]).reset_index(drop=True)

    grouped = long.groupby(["league", "team"], sort=False)
    for col in ("pts", "gd"):
        long[f"{col}_form"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).mean()
        )
    gap = (long["date"] - grouped["date"].shift(1)).dt.days
    stale = gap.isna() | (gap > max_gap_days)
    long.loc[stale, ["pts_form", "gd_form"]] = np.nan

    wide = long.pivot(index="match", columns="side", values=["pts_form", "gd_form"])
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    return wide.reindex(matches.index)


def v1_form(matches: pd.DataFrame, window: int = 5, include_current: bool = True) -> pd.DataFrame:
    """The original version's form feature, kept only to measure its leak.

    Mean goals of the home side over its last `window` HOME matches, and of the
    away side over its last `window` AWAY matches. With include_current=True
    (what version 1 did) the window includes the match being predicted, so the
    feature already contains part of the result. include_current=False is the
    same feature computed honestly, for comparison.
    """
    ordered = matches.sort_values("date")

    def roll(s: pd.Series) -> pd.Series:
        return (s if include_current else s.shift(1)).rolling(window).mean()

    home = ordered.groupby(["league", "home"])["home_goals"].transform(roll)
    away = ordered.groupby(["league", "away"])["away_goals"].transform(roll)
    return pd.DataFrame({"v1_home_form": home, "v1_away_form": away}).reindex(matches.index)


def market_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """Consensus pre-match probabilities from the bookmakers' 1X2 odds.

    For each bookmaker with all three prices, 1/odds is scaled to sum to 1
    (removing the margin); the consensus is the average across bookmakers.
    """
    probs = []
    for book in BOOKMAKERS:
        odds = matches[[f"{book}H", f"{book}D", f"{book}A"]].astype(float)
        valid = (odds > 1.0).all(axis=1)
        inv = 1.0 / odds.where(valid)
        probs.append((inv.div(inv.sum(axis=1), axis=0)).to_numpy())
    stacked = np.stack(probs)  # bookmaker x match x outcome
    n_books = np.isfinite(stacked[:, :, 0]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.nansum(stacked, axis=0) / n_books[:, None]  # matches with no odds stay empty
    out = pd.DataFrame(consensus, index=matches.index, columns=["mkt_H", "mkt_D", "mkt_A"])
    out["n_books"] = n_books
    return out


def lineup_ratings(matches: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Average FIFA rating of each starting XI, using each player's latest rating dated before the match."""
    cols = [f"{side}_player_{i}" for side in ("home", "away") for i in range(1, 12)]
    long = matches[["match_id", "date"] + cols].melt(id_vars=["match_id", "date"], var_name="slot", value_name="player_api_id")
    long = long.dropna(subset=["player_api_id"])
    long["player_api_id"] = long["player_api_id"].astype("int64")
    long["side"] = long["slot"].str.split("_").str[0]
    # A rating dated on match day may already reflect that match, so only strictly earlier dates count.
    long["lookup"] = long["date"] - pd.Timedelta(days=1)
    rated = pd.merge_asof(
        long.sort_values("lookup"),
        ratings.rename(columns={"date": "rating_date"}).sort_values("rating_date"),
        left_on="lookup", right_on="rating_date", by="player_api_id", direction="backward",
    )
    xi = rated.groupby(["match_id", "side"])["overall_rating"].agg(["mean", "count"]).unstack("side")
    xi.columns = [f"xi_{side}_{stat}" for stat, side in xi.columns]
    return matches[["match_id"]].join(xi, on="match_id").drop(columns="match_id")
