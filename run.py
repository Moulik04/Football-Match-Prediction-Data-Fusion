"""Rebuild every table in results/ from the data in data/.

    python run.py

Part 1 uses the Big Five results file in this repository and always runs.
Part 2 needs the Kaggle European Soccer Database (see data/README.md). It joins
those matches to the Big Five file, then adds bookmaker odds and starting
line-ups. It is skipped, with a note, if the database is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from matchfusion import data, features, names
from matchfusion import evaluate as ev

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Part 1: 1995/96 and 1996/97 only warm up the ratings; the test seasons are never seen in fitting or tuning.
TRAIN_1, TEST_1 = range(1997, 2016), range(2016, 2019)
TUNE_FIT, TUNE_CHECK = range(1997, 2013), range(2013, 2016)
# Part 2: the Kaggle database covers 2008/09 to 2015/16.
TRAIN_2, TEST_2 = range(2008, 2014), range(2014, 2016)

FORM_COLS = ["home_pts_form", "away_pts_form", "home_gd_form", "away_gd_form"]


def add_diffs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["pts_form_diff"] = (df["home_pts_form"] - df["away_pts_form"]).fillna(0.0)
    df["gd_form_diff"] = (df["home_gd_form"] - df["away_gd_form"]).fillna(0.0)
    return df


def home_advantage(bf: pd.DataFrame) -> float:
    """Home advantage in rating points, read off the training seasons' home record.

    A home side scores 1 for a win and 0.5 for a draw; if its average is E, the
    Elo gap that predicts E is 400 * log10(E / (1 - E)).
    """
    train = bf[bf["season"].isin(TRAIN_1)]
    e = np.where(train["result"] == "H", 1.0, np.where(train["result"] == "D", 0.5, 0.0)).mean()
    return float(400 * np.log10(e / (1 - e)))


def tune_elo(bf: pd.DataFrame, hfa: float) -> tuple[float, pd.DataFrame]:
    """Pick the Elo update size k on 2013/14 to 2015/16, fitting on earlier seasons only."""
    rows = []
    for k in (5, 10, 15, 20, 25, 30, 40):
        elo = features.elo_ratings(bf, k=k, home_adv=hfa)
        frame = bf.assign(elo_diff=elo["elo_home"] - elo["elo_away"])
        fit, check = ev.chronological_split(frame, TUNE_FIT, TUNE_CHECK)
        probs = ev.fit_predict(ev.logistic(), fit, check, ["elo_diff"])
        rows.append({"k": k, "check_log_loss": ev.scores(probs, check["result"])["log_loss"]})
    grid = pd.DataFrame(rows)
    return float(grid.loc[grid["check_log_loss"].idxmin(), "k"]), grid


def part1(bf: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    train, test = ev.chronological_split(bf, TRAIN_1, TEST_1)
    football = ["elo_diff", "pts_form_diff", "gd_form_diff"]
    raw = ["elo_home", "elo_away"] + FORM_COLS
    candidates = {
        "Base rates from the training seasons": ev.base_rates(train["result"], len(test)),
        "Elo only (logistic regression)": ev.fit_predict(ev.logistic(), train, test, ["elo_diff"]),
        "Elo + recent form (logistic regression)": ev.fit_predict(ev.logistic(), train, test, football),
        "Elo + recent form (gradient boosting)": ev.fit_predict(ev.boosted(), train, test, raw),
    }
    rows = [{"model": name, **ev.scores(p, test["result"])} for name, p in candidates.items()]
    info = {"train_matches": len(train), "test_matches": len(test),
            "test_seasons": "2016/17 to 2018/19", "train_seasons": "1997/98 to 2015/16"}
    return {"rows": rows, "info": info}, test.assign(**{"p_" + o: candidates["Elo + recent form (logistic regression)"][:, i] for i, o in enumerate(ev.OUTCOMES)}), candidates


def leak_check(df: pd.DataFrame, extra: list[str], train_seasons: range, test_seasons: range) -> list[dict]:
    """Version 1's form feature with and without the leak, with a random and a chronological split."""
    df = df.copy()
    leaky = features.v1_form(df, include_current=True)
    honest = features.v1_form(df, include_current=False)
    df["home_win"] = (df["result"] == "H").astype(int)
    rows = []
    setups = [
        ("Version 1: form includes the match, random 80/20 split", leaky, "random"),
        ("Form includes the match, earlier seasons predict later ones", leaky, "time"),
        ("Form from earlier matches only, earlier seasons predict later ones", honest, "time"),
    ]
    for label, form, split in setups:
        frame = df.join(form).dropna(subset=["v1_home_form", "v1_away_form"] + extra)
        cols = ["v1_home_form", "v1_away_form"] + extra
        frame = frame[frame["season"].isin(list(train_seasons) + list(test_seasons))]
        if split == "random":
            train, test = train_test_split(frame, test_size=0.2, stratify=frame["home_win"], random_state=42)
        else:
            train, test = frame[frame["season"].isin(train_seasons)], frame[frame["season"].isin(test_seasons)]
        model = ev.boosted()
        model.fit(train[cols].to_numpy(), train["home_win"].to_numpy())
        p = model.predict_proba(test[cols].to_numpy())[:, 1]
        from sklearn.metrics import roc_auc_score
        rows.append({"setup": label, "home_win_auc": float(roc_auc_score(test["home_win"], p)), "test_matches": len(test)})
    return rows


def part2(bf: pd.DataFrame, db_path: Path) -> dict:
    kg, ratings = data.load_kaggle(db_path)
    ref = bf[bf["season"].between(2008, 2015)]
    name_map = names.resolve_by_fixtures(kg, ref)
    joined = names.join_on_resolved_names(kg, bf, name_map)
    matched = joined["ref_index"].notna()
    agree = matched & (joined["ref_home_goals"] == joined["home_goals"]) & (joined["ref_away_goals"] == joined["away_goals"])

    fused = joined[matched].copy()
    fused["ref_index"] = fused["ref_index"].astype(int)
    fused = fused.join(bf.loc[fused["ref_index"], ["elo_home", "elo_away"] + FORM_COLS].set_index(fused.index))
    fused = add_diffs(fused)
    fused = fused.join(features.market_probabilities(fused)).join(features.lineup_ratings(fused, ratings))
    fused["xi_diff"] = fused["xi_home_mean"] - fused["xi_away_mean"]
    fused["mkt_logit_ha"] = np.log(fused["mkt_H"] / fused["mkt_A"])
    fused["mkt_logit_d"] = np.log(fused["mkt_D"] / (1 - fused["mkt_D"]))
    usable = fused.dropna(subset=["mkt_H", "xi_diff", "elo_diff"])
    usable = usable[(usable["xi_home_count"] >= 9) & (usable["xi_away_count"] >= 9)]

    train, test = ev.chronological_split(usable, TRAIN_2, TEST_2)
    football = ["elo_diff", "pts_form_diff", "gd_form_diff"]
    market = ["mkt_logit_ha", "mkt_logit_d"]
    candidates = {
        "Base rates from the training seasons": ev.base_rates(train["result"], len(test)),
        "Bookmakers' consensus odds (no model)": test[["mkt_H", "mkt_D", "mkt_A"]].to_numpy(),
        "Elo + recent form (logistic regression)": ev.fit_predict(ev.logistic(), train, test, football),
        "Elo + form + line-up ratings (logistic regression)": ev.fit_predict(ev.logistic(), train, test, football + ["xi_diff"]),
        "Odds + Elo + form + line-ups (logistic regression)": ev.fit_predict(ev.logistic(), train, test, market + football + ["xi_diff"]),
        "Odds + Elo + form + line-ups (gradient boosting)": ev.fit_predict(
            ev.boosted(), train, test, ["mkt_H", "mkt_D", "mkt_A", "elo_home", "elo_away", "xi_home_mean", "xi_away_mean"] + FORM_COLS),
    }
    rows = [{"model": name, **ev.scores(p, test["result"])} for name, p in candidates.items()]
    vs_market = ev.paired_bootstrap(candidates["Odds + Elo + form + line-ups (logistic regression)"],
                                    candidates["Bookmakers' consensus odds (no model)"], test["result"])

    leak = leak_check(fused.assign(b365_H=1 / fused["B365H"], b365_D=1 / fused["B365D"], b365_A=1 / fused["B365A"]),
                      ["b365_H", "b365_D", "b365_A"], TRAIN_2, TEST_2)

    disagreements = joined[matched & ~agree][["date", "league", "home", "away", "home_goals", "away_goals", "ref_home_goals", "ref_away_goals"]]
    report = {
        "kaggle_matches": int(len(kg)),
        "kaggle_clubs": int(len(name_map)),
        "clubs_matched": int(name_map["ref_name"].notna().sum()),
        "name_clashes": int(name_map["clash"].sum()),
        "clubs_named_differently": int((name_map["ref_name"].notna() & (name_map["src_name"] != name_map["ref_name"])).sum()),
        "matches_joined": int(matched.sum()),
        "join_rate": float(matched.mean()),
        "score_agreement": float(agree.sum() / matched.sum()),
        "score_disagreements": int((matched & ~agree).sum()),
        "spelling_only_errors": names.spelling_only_errors(name_map),
        "usable_matches": int(len(usable)),
        "train_matches": int(len(train)),
        "test_matches": int(len(test)),
        "test_seasons": "2014/15 and 2015/16",
        "train_seasons": "2008/09 to 2013/14",
        "fused_vs_odds_log_loss": vs_market,
    }
    return {"rows": rows, "leak": leak, "name_map": name_map, "disagreements": disagreements,
            "report": report, "test": test.assign(**{f"p_fused_{o}": candidates["Odds + Elo + form + line-ups (logistic regression)"][:, i] for i, o in enumerate(ev.OUTCOMES)})}


def calibration_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle="--", label="Perfect calibration")
    for (label, (p, y)), color in zip(curves.items(), ["#1f77b4", "#d62728", "#2ca02c"]):
        bins = np.clip((p * 10).astype(int), 0, 9)
        frame = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        frame = frame[frame["n"] >= 30]
        ax.plot(frame["p"], frame["y"], marker="o", color=color, label=label)
    ax.set_xlabel("Predicted chance of a home win")
    ax.set_ylabel("Share of those matches the home side won")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Home-win calibration on the test seasons", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    bf = data.load_big_five()
    hfa = home_advantage(bf)
    k, grid = tune_elo(bf, hfa)
    grid.round(6).to_csv(RESULTS / "elo_tuning.csv", index=False)
    bf = add_diffs(bf.join(features.elo_ratings(bf, k=k, home_adv=hfa)).join(features.rolling_form(bf)))

    p1, test1, _ = part1(bf)
    leak1 = leak_check(bf, [], TRAIN_1, TEST_1)
    summary = {"elo_k": k, "elo_home_advantage": hfa, "big_five_matches": len(bf),
               "big_five_seasons": "1995/96 to 2018/19", "part1": p1, "leak_check_big_five": leak1}
    md = ["# Results", "",
          f"Generated by `python run.py`. Elo home advantage = {hfa:.0f} rating points, read off the training seasons' "
          f"home record; update size k = {k:g}, chosen on 2013/14 to 2015/16 using earlier seasons only (see elo_tuning.csv).", "",
          f"## Part 1: Big Five results only ({p1['info']['test_matches']:,} test matches, {p1['info']['test_seasons']})", "",
          f"Trained on {p1['info']['train_matches']:,} matches from {p1['info']['train_seasons']}. Lower is better for log loss, RPS and Brier.", "",
          ev.table(p1["rows"]), "",
          "## Leak check on the Big Five file (version 1's form feature, no odds)", "",
          ev.table(leak1, digits=3), ""]
    curves = {"Elo + form, Big Five 2016/17 to 2018/19": (test1["p_H"].to_numpy(), (test1["result"] == "H").to_numpy(dtype=float))}

    db = data.find_kaggle_db()
    if db is None:
        md += ["## Part 2: odds and line-ups", "",
               "Skipped: the Kaggle European Soccer Database is not in data/. See data/README.md.", ""]
        summary["part2"] = None
    else:
        p2 = part2(bf, db)
        p2["name_map"].to_csv(RESULTS / "name_matching.csv", index=False)
        p2["disagreements"].to_csv(RESULTS / "score_disagreements.csv", index=False)
        r = p2["report"]
        summary["part2"] = {"rows": p2["rows"], "leak": p2["leak"], "report": r}
        md += [f"## Part 2: adding bookmaker odds and starting line-ups ({r['test_matches']:,} test matches, {r['test_seasons']})", "",
               f"Trained on {r['train_matches']:,} matches from {r['train_seasons']}. "
               "Every model is scored on the same matches.", "",
               ev.table(p2["rows"]), "",
               "Odds + Elo + form + line-ups (logistic regression) minus the odds alone, per-match log loss: "
               f"{r['fused_vs_odds_log_loss']['mean_diff']:+.4f} (95% bootstrap interval "
               f"{r['fused_vs_odds_log_loss']['low']:+.4f} to {r['fused_vs_odds_log_loss']['high']:+.4f}).", "",
               "### Joining the two sources", "",
               f"- Kaggle matches in the five leagues: {r['kaggle_matches']:,}; clubs: {r['kaggle_clubs']}, "
               f"of which {r['clubs_named_differently']} are spelled differently in the Big Five file.",
               f"- Clubs matched by fixtures: {r['clubs_matched']} of {r['kaggle_clubs']} (name clashes: {r['name_clashes']}). "
               "Every pair is in name_matching.csv with its votes and string similarity.",
               f"- Matches joined: {r['matches_joined']:,} ({r['join_rate']:.1%}); the two sources agree on the score in "
               f"{r['score_agreement']:.2%} of them" + (f" ({r['score_disagreements']} disagreements, listed in score_disagreements.csv)." if r['score_disagreements'] else "."),
               f"- Matching on spelling alone (closest normalised name in the same league) would have paired "
               f"{len(r['spelling_only_errors'])} clubs wrongly: "
               + "; ".join(f"{e['src_name']} with {e['spelling_pick']} instead of {e['ref_name']}" for e in r['spelling_only_errors']) + ".",
               f"- Matches with odds and at least 9 rated starters per side: {r['usable_matches']:,}.", "",
               "### Leak check with odds (version 1's set-up: form + Bet365 odds, home win or not)", "",
               ev.table(p2["leak"], digits=3), ""]
        t2 = p2["test"]
        curves["Odds + Elo + form + line-ups, 2014/15 to 2015/16"] = (t2["p_fused_H"].to_numpy(), (t2["result"] == "H").to_numpy(dtype=float))
        curves["Bookmakers' odds alone, 2014/15 to 2015/16"] = (t2["mkt_H"].to_numpy(), (t2["result"] == "H").to_numpy(dtype=float))

    calibration_plot(curves, RESULTS / "calibration.png")
    (RESULTS / "RESULTS.md").write_text("\n".join(md))
    def rounded(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value

    (RESULTS / "summary.json").write_text(json.dumps(rounded(summary), indent=2, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
