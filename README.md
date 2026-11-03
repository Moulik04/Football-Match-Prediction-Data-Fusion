# Predicting Football Match Outcomes: Elo, Form and Data Fusion

Predicts home win, draw or away win for matches in Europe's top five leagues, using only what was known before kick-off, and scores every model on later seasons it never saw.

**This is version 2, a rebuild.** Version 1 reported a ROC-AUC of 0.78, but its form feature included the match being predicted and its test set was a random sample. Rerunning version 1's set-up gives 0.784; fixing the leak gives 0.707, about the same as the bookmakers' odds on their own (0.708). [`legacy/README.md`](legacy/README.md) explains what was wrong.

## Results

### Part 1: Elo and form on 24 seasons of results

Trained on 33,806 matches (1997/98 to 2015/16) and scored on all 5,478 matches from 2016/17 to 2018/19:

| Model | Log loss | RPS | Brier | Accuracy | Home-win ROC-AUC |
|---|---|---|---|---|---|
| Base rates from the training seasons | 1.063 | 0.228 | 0.642 | 46.2% | 0.500 |
| Elo only (logistic regression) | 0.967 | 0.196 | 0.574 | 54.0% | 0.711 |
| Elo + recent form (logistic regression) | 0.966 | 0.196 | 0.574 | 54.1% | 0.712 |
| Elo + recent form (gradient boosting) | 0.967 | 0.196 | 0.574 | 53.8% | 0.712 |

Log loss, RPS (ranked probability score) and Brier: lower is better.

- **Elo does almost all the work.** Ratings built from the results alone take log loss from 1.063 to 0.967.
- **Recent form adds almost nothing once Elo is in** (0.967 to 0.966). A club's last five results are already in its rating.
- **Gradient boosting doesn't beat logistic regression** on these few, smooth features.
- **The probabilities are well calibrated:** when the model gives the home side a 60% to 70% chance, the home side wins about 65% of the time ([`results/calibration.png`](results/calibration.png)).

### Part 2: against the bookmakers, with starting line-ups

The Kaggle European Soccer Database adds 1X2 odds and starting line-ups for 2008/09 to 2015/16. Trained on 10,920 matches (2008/09 to 2013/14) and scored on 3,651 matches from 2014/15 and 2015/16, every model on the same matches:

| Model | Log loss | RPS | Brier | Accuracy | Home-win ROC-AUC |
|---|---|---|---|---|---|
| Base rates from the training seasons | 1.071 | 0.228 | 0.648 | 44.7% | 0.500 |
| Bookmakers' consensus odds (no model) | 0.980 | 0.198 | 0.584 | 52.1% | 0.708 |
| Elo + recent form (logistic regression) | 0.994 | 0.202 | 0.594 | 51.1% | 0.693 |
| Elo + form + line-up ratings (logistic regression) | 0.992 | 0.201 | 0.592 | 51.4% | 0.696 |
| Odds + Elo + form + line-ups (logistic regression) | 0.980 | 0.198 | 0.584 | 51.9% | 0.709 |
| Odds + Elo + form + line-ups (gradient boosting) | 0.986 | 0.199 | 0.587 | 51.9% | 0.705 |

- **The bookmakers are the benchmark.** Their consensus odds (0.980) beat every model built without them; the best of those reaches 0.992.
- **Line-ups add real information.** The average FIFA rating of each starting XI, using ratings dated before the match, improves Elo + form from 0.994 to 0.992.
- **Adding everything to the odds doesn't beat them.** The combined model scores 0.9795 against 0.9803 for the odds alone, and the 95% bootstrap interval for the difference (−0.0032 to +0.0017) includes zero. The market already prices form, ratings and line-ups.

### Leak check: version 1's set-up

Version 1 averaged each club's goals over its last five home (or away) matches, including the match being predicted. The same model with and without that leak:

| Set-up (home win or not) | Big Five file, no odds | Kaggle data with Bet365 odds (version 1's set-up) |
|---|---|---|
| Version 1: form includes the match, random 80/20 split | 0.762 | 0.784 |
| Form includes the match, earlier seasons predict later ones | 0.783 | 0.773 |
| Form from earlier matches only, earlier seasons predict later ones | 0.639 | 0.707 |

The leak accounts for the gap between 0.78 and 0.71.

## Data and name matching

| Source | What it adds | How it names clubs |
|---|---|---|
| [The Big Five European soccer leagues data](https://www.openml.org/search?type=data&id=43535) (in `data/`, public domain, from openfootball) | Results for 1995/96 to 2018/19, the basis for Elo and form | "Bayern München", "Manchester United FC", "SS Robur Siena" |
| [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) (Kaggle, download separately) | 1X2 odds from six bookmakers, starting line-ups, dated FIFA player ratings | "FC Bayern Munich", "Manchester United", "Siena" |

80 of the 164 clubs in both sources are spelled differently, so [`matchfusion/names.py`](matchfusion/names.py) matches clubs by the fixtures they played rather than by spelling. Matches in the same league, on the same date or a day apart, with the same score, vote for "this club is that club"; the true pair collects hundreds of votes and beats the runner-up at least sevenfold.

- All 164 clubs matched with no clashes, all 14,585 Kaggle matches in the five leagues joined, and the two sources agree on every score.
- Matching on spelling alone would have paired three clubs wrongly: Siena with Cesena, Inter with Udinese and Reggio Calabria with Cagliari.
- Every pair, with its votes and string similarity, is in [`results/name_matching.csv`](results/name_matching.csv).

## Method

- **Only pre-match information.** Elo ratings are taken before each match is played. Form is points and goal difference per game over each club's previous five matches, with the current match dropped before averaging. Line-up ratings use each player's latest FIFA rating dated before match day. [`tests/test_leakage.py`](tests/test_leakage.py) changes a match's score and checks that its own features don't move.
- **Elo.** Computed league by league from the results. The home advantage (72 rating points) is read off the training seasons' home record; the update size (k = 15) was chosen on 2013/14 to 2015/16, fitting on earlier seasons only ([`results/elo_tuning.csv`](results/elo_tuning.csv)). A promoted club starts at the average rating of the clubs that went down the season before. Bigger wins move ratings more.
- **Odds.** For each bookmaker, 1 / odds is scaled to sum to 1 to remove the margin; the consensus is the average across bookmakers.
- **Split by season, never at random.** No test season is used for fitting or tuning.
- **Scores.** Log loss and Brier score judge the probabilities; RPS also credits a near miss, such as predicting a draw when the home side wins; accuracy and home-win ROC-AUC are there for comparison with version 1.

## Limitations

- Chance quality isn't used: neither source has expected goals, and this version doesn't parse the Kaggle file's shot events.
- One fixed split per part. A rolling season-by-season evaluation would show how stable these numbers are.
- FIFA ratings are a video game's judgement of players, updated a few times a season.
- Elo ignores transfers and managerial changes until results reflect them.

## Run it

```bash
pip install -r requirements.txt
python run.py                        # writes results/ (Part 2 needs the Kaggle file, see data/README.md)
python -m unittest discover -s tests
```

Tested with Python 3.13, pandas 3.0, scikit-learn 1.9 and matplotlib 3.11.
