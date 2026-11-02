"""Checks that no feature can see the result it is used to predict.

    python -m unittest discover -s tests
"""
import unittest

import numpy as np
import pandas as pd

from matchfusion import data, evaluate, features, names


def season(scores, league="Test League", start="2010-08-01"):
    """A toy season in which clubs A and B play each other on consecutive weeks."""
    rows = []
    for i, (hg, ag) in enumerate(scores):
        home, away = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append({"league": league, "season": 2010, "date": pd.Timestamp(start) + pd.Timedelta(weeks=i),
                     "home": home, "away": away, "home_goals": hg, "away_goals": ag})
    df = pd.DataFrame(rows)
    df["result"] = data.result_from_goals(df["home_goals"], df["away_goals"])
    return df


class EloTest(unittest.TestCase):
    def test_ratings_are_taken_before_the_match(self):
        base = season([(1, 0), (0, 0), (2, 1)])
        changed = season([(1, 0), (5, 0), (2, 1)])
        a, b = features.elo_ratings(base), features.elo_ratings(changed)
        self.assertEqual(a.loc[0, "elo_home"], 1500.0)
        # Match 2's own ratings ignore match 2's score; match 3's ratings do not.
        pd.testing.assert_series_equal(a.loc[1], b.loc[1])
        self.assertNotEqual(a.loc[2, "elo_home"], b.loc[2, "elo_home"])


class FormTest(unittest.TestCase):
    def test_form_excludes_the_current_match(self):
        scores = [(1, 0), (0, 1), (2, 2), (1, 1), (0, 0), (3, 0), (1, 2)]
        base, changed = season(scores), season(scores[:-1] + [(9, 0)])
        a, b = features.rolling_form(base), features.rolling_form(changed)
        pd.testing.assert_frame_equal(a, b)  # only the last match changed, and nothing comes after it

    def test_form_uses_earlier_matches(self):
        scores = [(1, 0), (0, 1), (2, 2), (1, 1), (0, 0), (3, 0), (1, 2)]
        base, changed = season(scores), season(scores[:4] + [(9, 0)] + scores[5:])
        a, b = features.rolling_form(base), features.rolling_form(changed)
        self.assertFalse(a.loc[5].equals(b.loc[5]))

    def test_version_1_form_leaks(self):
        scores = [(1, 0), (0, 1), (2, 2), (1, 1), (0, 0), (3, 0), (1, 2), (0, 0), (2, 0), (1, 1), (4, 0), (0, 0)]
        base, changed = season(scores), season(scores[:-2] + [(9, 0), (0, 0)])
        leaky = features.v1_form(base, include_current=True).loc[10, "v1_home_form"]
        self.assertNotEqual(leaky, features.v1_form(changed, include_current=True).loc[10, "v1_home_form"])
        honest = features.v1_form(base, include_current=False).loc[10, "v1_home_form"]
        self.assertEqual(honest, features.v1_form(changed, include_current=False).loc[10, "v1_home_form"])


class SplitTest(unittest.TestCase):
    def test_training_ends_before_testing(self):
        bf = data.load_big_five()
        train, test = evaluate.chronological_split(bf, range(1997, 2016), range(2016, 2019))
        self.assertLess(train["date"].max(), test["date"].min())


class NamesTest(unittest.TestCase):
    def test_fixtures_beat_spelling(self):
        ref = season([(1, 0), (0, 2), (3, 3), (1, 1)])
        src = ref.replace({"A": "Club Alpha FC", "B": "Beta United"})
        table = names.resolve_by_fixtures(src, ref).set_index("src_name")
        self.assertEqual(table.loc["Club Alpha FC", "ref_name"], "A")
        self.assertEqual(table.loc["Beta United", "ref_name"], "B")


class MarketTest(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        odds = {f"{b}{o}": [np.nan] * 2 for b in data.BOOKMAKERS for o in "HDA"}
        odds.update({"B365H": [2.0, 1.5], "B365D": [3.4, 4.0], "B365A": [4.0, 7.0]})
        probs = features.market_probabilities(pd.DataFrame(odds))
        np.testing.assert_allclose(probs[["mkt_H", "mkt_D", "mkt_A"]].sum(axis=1), 1.0)
        self.assertEqual(list(probs["n_books"]), [1, 1])


if __name__ == "__main__":
    unittest.main()
