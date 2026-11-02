# Version 1

The original code, report and slides, unchanged, so the history stays visible. Don't rely on their numbers. Version 2, the rest of this repository, fixes three problems:

1. **The form feature leaked the result.** It was a 5-match rolling mean of goals that included the match being predicted. Rerunning version 1's set-up on its own data (the Kaggle matches with Bet365 odds) gives a home-win ROC-AUC of 0.784; computing form from earlier matches only, with a season split, gives 0.707 (see `results/RESULTS.md`). The reported 0.78 came mostly from the leak.
2. **The split was random.** 80% of matches were drawn at random for training, so the model learned from games played after the ones it was tested on.
3. **The write-up claimed work the code doesn't do.** The report and slides describe FBref player statistics, team names harmonized across sources and a 10% gain from fusing sources. The version 1 code has none of these. It joins the Kaggle match table to an odds file on exact team names, computes player ratings it never uses, and fits on two goals-based form columns plus Bet365's odds.
