# Data and source notes

## Chosen dataset

The reproduction uses `data/raw/ultimate_ufc_dataset.csv`, from the TidyTuesday UFC release dated 2026-07-07. Its dataset documentation describes Red/Blue corner labels, outcome, dated fighter statistics, fighter age, and multiple pre-bout career totals. It covers 2010-03-21 through 2026-03-28. For comparison with the reference study, this project restricts modeling to bouts dated 2010-01-01 through 2019-12-31. The chosen slice has 4,073 labeled bouts and contains no draws, so the measured classification task is binary.

The paper's Google Drive folder URL returned a 404, and Kaggle's original 1993–2021 dataset download required an account. The TidyTuesday mirror is publicly downloadable and preserves pre-fight career features, making it suitable for a documented approximation.

## Feature policy

Included: historical fight counts and streaks, pre-fight career rates, fighter age at the bout, height, reach, listed weigh-in weight, and stance for each corner.

Excluded: fighter names, event date/location/country, weight-class and title context, fight duration/round/method, rankings, betting odds, finish labels, and other outcome-derived fields. The official paper excluded event circumstances; this project also excludes those fields to keep the task fighter-centric.

## Difference from the paper

The paper used 3,355 complete records across 1997–2019 with 134 features. The replacement begins in 2010 and has 118 columns before this project selects 48 raw features. It is not the paper's original dataset. Results are compared as an approximate reproduction only.
