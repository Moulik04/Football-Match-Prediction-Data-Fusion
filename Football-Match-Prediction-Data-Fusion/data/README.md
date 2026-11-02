# Data

## In this folder

`BIG_FIVE_1995-2019.csv`: every top-division match in England, Spain, Germany, Italy and France from 1995/96 to March 2020 (44,269 matches). It is "The Big Five European soccer leagues data" ([OpenML 43535](https://www.openml.org/search?type=data&id=43535)), built from the openfootball [football.db](https://openfootball.github.io/) project and released into the public domain (CC0). `run.py` uses 1995/96 to 2018/19 and drops 2019/20, which stops in March 2020.

## Download separately (Part 2)

The European Soccer Database by Hugo Mathien: [kaggle.com/datasets/hugomathien/soccer](https://www.kaggle.com/datasets/hugomathien/soccer). Download it (a free Kaggle account is needed) and put the zip, or the `database.sqlite` inside it, in this folder. It isn't committed here because it is about 300 MB unzipped.

It has 25,979 matches from 11 European leagues, 2008/09 to 2015/16, with starting line-ups, 1X2 odds from several bookmakers and dated EA Sports FIFA player ratings. Part 2 uses the 14,585 matches from the five leagues that the Big Five file also covers.

The two sources spell many clubs differently ("Bayern München" and "FC Bayern Munich", "Manchester United FC" and "Manchester United"), so `matchfusion/names.py` matches clubs by the fixtures they played rather than by spelling. The pairs it found are in `results/name_matching.csv`.
