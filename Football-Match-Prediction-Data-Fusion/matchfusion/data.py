"""Load the match sources into one shape.

Every loader returns one row per match with the columns
league, season (start year, so 2015 = 2015/16), date, home, away,
home_goals, away_goals and result (H, D or A).

Team names are kept exactly as each source spells them. names.py
reconciles them; nothing here renames a club.
"""
from __future__ import annotations

import sqlite3
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Country codes used by the Big Five file.
BIG_FIVE_LEAGUES = {
    "ENG": "Premier League",
    "ESP": "La Liga",
    "GER": "Bundesliga",
    "IT": "Serie A",
    "FR": "Ligue 1",
}

# League names used by the Kaggle European Soccer Database.
KAGGLE_LEAGUES = {
    "England Premier League": "Premier League",
    "Spain LIGA BBVA": "La Liga",
    "Germany 1. Bundesliga": "Bundesliga",
    "Italy Serie A": "Serie A",
    "France Ligue 1": "Ligue 1",
}

# Bookmakers whose 1X2 odds the Kaggle database carries.
BOOKMAKERS = ["B365", "BW", "IW", "LB", "WH", "VC"]


def result_from_goals(home_goals, away_goals) -> np.ndarray:
    return np.where(home_goals > away_goals, "H", np.where(home_goals < away_goals, "A", "D"))


def load_big_five(path: Path = DATA_DIR / "BIG_FIVE_1995-2019.csv", last_season: int = 2018) -> pd.DataFrame:
    """Results for the top divisions of England, Spain, Germany, Italy and France.

    Source: "The Big Five European soccer leagues data" (OpenML 43535),
    built from the openfootball football.db project (public domain).
    The 2019/20 season stops in March 2020 because the file was collected
    in April 2020, so it is dropped by default.
    """
    raw = pd.read_csv(path)
    date = raw["Date"].str.extract(r"(\d{1,2} \w{3} \d{4})")[0]
    df = pd.DataFrame(
        {
            "league": raw["Country"].map(BIG_FIVE_LEAGUES),
            "season": raw["Year"].astype(int),
            "date": pd.to_datetime(date, format="%d %b %Y"),
            "home": raw["Team 1"].str.strip(),
            "away": raw["Team 2"].str.strip(),
            "home_goals": raw["FT Team 1"].astype(int),
            "away_goals": raw["FT Team 2"].astype(int),
        }
    )
    if df["league"].isna().any():
        raise ValueError("unknown country code in the Big Five file")
    df = df[df["season"] <= last_season].copy()
    df["result"] = result_from_goals(df["home_goals"], df["away_goals"])
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


def find_kaggle_db(search_dirs=(DATA_DIR,)) -> Path | None:
    """Return the Kaggle database.sqlite, unzipping it from a downloaded zip if needed."""
    for folder in map(Path, search_dirs):
        direct = folder / "database.sqlite"
        if direct.exists():
            return direct
        for zpath in sorted(folder.glob("*.zip")):
            with zipfile.ZipFile(zpath) as zf:
                members = [m for m in zf.namelist() if m.endswith("database.sqlite")]
                if members:
                    target = Path(tempfile.gettempdir()) / "matchfusion_database.sqlite"
                    if not target.exists():
                        with zf.open(members[0]) as src, open(target, "wb") as dst:
                            while chunk := src.read(1 << 20):
                                dst.write(chunk)
                    return target
    return None


def load_kaggle(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matches (with starting line-ups and 1X2 odds) and dated player ratings.

    Source: "European Soccer Database" by Hugo Mathien on Kaggle
    (kaggle.com/datasets/hugomathien/soccer), seasons 2008/09 to 2015/16.
    Player ratings are EA Sports FIFA ratings as scraped from sofifa,
    one row per rating update, each with the date it took effect.
    """
    lineup_cols = [f"{side}_player_{i}" for side in ("home", "away") for i in range(1, 12)]
    odds_cols = [f"{b}{o}" for b in BOOKMAKERS for o in "HDA"]
    query = f"""
        SELECT m.match_api_id AS match_id, m.date, m.season, l.name AS league_name,
               th.team_long_name AS home, ta.team_long_name AS away,
               m.home_team_goal AS home_goals, m.away_team_goal AS away_goals,
               {", ".join("m." + c for c in lineup_cols + odds_cols)}
        FROM Match m
        JOIN League l ON l.id = m.league_id
        JOIN Team th ON th.team_api_id = m.home_team_api_id
        JOIN Team ta ON ta.team_api_id = m.away_team_api_id
    """
    with sqlite3.connect(db_path) as con:
        matches = pd.read_sql(query, con)
        ratings = pd.read_sql(
            "SELECT player_api_id, date, overall_rating FROM Player_Attributes "
            "WHERE overall_rating IS NOT NULL",
            con,
        )
    matches["league"] = matches.pop("league_name").map(KAGGLE_LEAGUES)
    matches = matches[matches["league"].notna()].copy()
    matches["season"] = matches["season"].str[:4].astype(int)
    matches["date"] = pd.to_datetime(matches["date"]).dt.normalize()
    matches["result"] = result_from_goals(matches["home_goals"], matches["away_goals"])
    ratings["date"] = pd.to_datetime(ratings["date"]).dt.normalize()
    return matches.sort_values(["date", "league", "home"]).reset_index(drop=True), ratings
