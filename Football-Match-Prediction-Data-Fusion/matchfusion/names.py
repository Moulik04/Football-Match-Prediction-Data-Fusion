"""Match club names across sources that spell them differently.

The Big Five file says "Bayern München" and "Manchester United FC"; the Kaggle
database says "FC Bayern Munich" and "Manchester United". String similarity
alone gets some of these wrong, so names are matched by the fixtures they
played: two sources' matches in the same league on the same date (or a day
apart) with the same score vote for "this home club is that home club" and
"this away club is that away club". The true pair collects hundreds of
votes; coincidences (two different 1-0 wins on the same Saturday) collect a
handful. String similarity is reported next to every pair as a second check.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

# Words that carry no identity ("FC", "Club", founding years and so on).
_FILLER = {
    "fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "us", "sv", "vfb", "vfl", "tsg", "fsv", "bv",
    "rc", "rcd", "cd", "ud", "sd", "ca", "club", "de", "del", "la", "calcio", "hsc", "ogc", "sco",
    "olympique", "stade", "real", "1", "04", "05", "96", "1846", "1860", "1899", "1900", "1907", "1909",
}


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(t for t in text.split() if t not in _FILLER)


def similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, normalise(a), normalise(b)).ratio(), 2)


def resolve_by_fixtures(src: pd.DataFrame, ref: pd.DataFrame, max_day_gap: int = 1) -> pd.DataFrame:
    """Map every club name in `src` to a club name in `ref`, league by league.

    Returns one row per `src` name: the chosen `ref` name, its votes, the share
    of that club's matches that voted for it, the runner-up's votes and the
    string similarity of the two names.
    """
    keys = ["league", "home_goals", "away_goals"]
    pairs = []
    for gap in range(-max_day_gap, max_day_gap + 1):
        shifted = ref[keys + ["date", "home", "away"]].assign(date=ref["date"] + pd.Timedelta(days=gap))
        hit = src[keys + ["date", "home", "away"]].merge(shifted, on=keys + ["date"], suffixes=("_src", "_ref"))
        pairs.append(hit)
    hits = pd.concat(pairs, ignore_index=True)
    votes = pd.concat([
        hits[["league", "home_src", "home_ref"]].set_axis(["league", "src_name", "ref_name"], axis=1),
        hits[["league", "away_src", "away_ref"]].set_axis(["league", "src_name", "ref_name"], axis=1),
    ]).value_counts().rename("votes").reset_index()

    played = pd.concat([src[["league", "home"]].set_axis(["league", "src_name"], axis=1),
                        src[["league", "away"]].set_axis(["league", "src_name"], axis=1)]).value_counts().rename("matches")

    rows = []
    for (league, name), group in votes.groupby(["league", "src_name"]):
        ranked = group.sort_values("votes", ascending=False)
        best = ranked.iloc[0]
        runner_up = int(ranked.iloc[1]["votes"]) if len(ranked) > 1 else 0
        rows.append({
            "league": league,
            "src_name": name,
            "ref_name": best["ref_name"],
            "votes": int(best["votes"]),
            "matches": int(played[(league, name)]),
            "runner_up_votes": runner_up,
            "similarity": similarity(name, best["ref_name"]),
        })
    table = pd.DataFrame(rows)
    table["vote_share"] = (table["votes"] / table["matches"]).round(3)
    # Two source names must never claim the same reference club.
    table["clash"] = table.duplicated(["league", "ref_name"], keep=False)
    unmatched = sorted(set(played.index) - set(zip(table["league"], table["src_name"])))
    for league, name in unmatched:
        table.loc[len(table)] = {"league": league, "src_name": name, "ref_name": None, "votes": 0,
                                 "matches": int(played[(league, name)]), "runner_up_votes": 0,
                                 "similarity": 0.0, "vote_share": 0.0, "clash": False}
    return table.sort_values(["league", "src_name"]).reset_index(drop=True)


def join_on_resolved_names(src: pd.DataFrame, ref: pd.DataFrame, name_map: pd.DataFrame, max_day_gap: int = 1) -> pd.DataFrame:
    """Join `src` matches to `ref` matches after renaming `src` clubs to `ref` names.

    Each pairing of clubs is played once per season at each ground, so league,
    season, home and away identify a match; the date is then checked to be
    within `max_day_gap` days. Returns `src` rows with ref_index (the matching
    `ref` row, or missing) and the reference score for comparison.
    """
    mapping = name_map.dropna(subset=["ref_name"]).set_index(["league", "src_name"])["ref_name"]
    renamed = src.copy()
    renamed["home_ref"] = [mapping.get((lg, n)) for lg, n in zip(src["league"], src["home"])]
    renamed["away_ref"] = [mapping.get((lg, n)) for lg, n in zip(src["league"], src["away"])]
    ref_keyed = ref.reset_index(names="ref_index")[["ref_index", "league", "season", "home", "away", "date", "home_goals", "away_goals"]]
    joined = renamed.merge(
        ref_keyed.rename(columns={"home": "home_ref", "away": "away_ref", "date": "ref_date",
                                  "home_goals": "ref_home_goals", "away_goals": "ref_away_goals"}),
        on=["league", "season", "home_ref", "away_ref"], how="left",
    )
    too_far = (joined["date"] - joined["ref_date"]).abs().dt.days > max_day_gap
    joined.loc[too_far, ["ref_index", "ref_home_goals", "ref_away_goals"]] = pd.NA
    joined.index = src.index
    return joined


def spelling_only_errors(name_map: pd.DataFrame) -> list[dict]:
    """Clubs that string similarity alone would pair wrongly, judged against the fixture-based match.

    For each source name, pick the reference club in the same league with the
    most similar normalised name, and list the cases where that pick differs.
    """
    errors = []
    for _, group in name_map.dropna(subset=["ref_name"]).groupby("league"):
        refs = group["ref_name"].tolist()
        for row in group.itertuples():
            pick = max(refs, key=lambda ref: SequenceMatcher(None, normalise(row.src_name), normalise(ref)).ratio())
            if pick != row.ref_name:
                errors.append({"league": row.league, "src_name": row.src_name,
                               "spelling_pick": pick, "ref_name": row.ref_name})
    return errors
