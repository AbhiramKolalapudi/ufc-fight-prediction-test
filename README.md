# UFC Fight Outcome Prediction

This project reproduces the four model families from McKinley McQuaide's 2019 CS229 report and adds a Random Forest model. It uses pre-fight fighter statistics to predict the winning corner. Although the source includes a Draw label, the 2010–2019 slice used here contains no draws, so the measured task is binary (Red or Blue).

## Dataset

The original report's Google Drive data link is no longer available. This project uses the public TidyTuesday UFC dataset curated by Benjamin Smith from the `fightr` R package. The bundled source file is `data/raw/ultimate_ufc_dataset.csv`. We keep bouts from 2010 through 2019 to align the end date with the paper. This source covers fewer earlier years than the paper's 1993–2019 data and has 118 columns overall.

The modeling script selects 48 fighter-centric fields: 46 numeric fields (23 for each corner) and stance for each corner. It excludes names, event location, rankings, betting odds, title and round context, and all post-fight fields. The dataset documentation says its career totals and streaks are strictly prior to each bout. Missing numeric fields are imputed from the training fold. No-contests are excluded.

## Reproduce

Use Python 3.10 or later.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\train.py
```

Outputs are saved in `results/`:

- `model_comparison.csv`: mean training and test accuracy, test accuracy standard deviation, and macro-F1 across the 20 chronological folds.
- `fold_metrics.csv`: per-fold metrics for all five models.
- `demo_predictions.csv`: last-fold examples predicted by Gradient Boosting.
- `run_metadata.json`: dataset period, label counts, feature list sizes, and split details.
- `results_summary.md`: measured results and the reported-versus-reproduced comparison.

## Model notebooks

Install the project dependencies, then open the project folder in VS Code. Install the Microsoft Python and Jupyter extensions if VS Code prompts you to. Open a notebook in `notebooks/`, choose the `.venv` Python kernel when prompted, and run its cells.

```powershell
python -m pip install -r requirements.txt
```

There is one notebook per model in `notebooks/`. Each notebook explains the model, shows the saved summary and fold-by-fold results, plots test accuracy over time, and includes cells to rerun that model using the same data and 20 time-series folds. To regenerate the notebooks and their charts after rerunning the main training script, use:

```powershell
python src\create_model_notebooks.py
```

## Models

| Role | Model | Paper settings reproduced |
|---|---|---|
| Reproduction | SGD classifier | Perceptron loss |
| Reproduction | MLP | Two hidden layers (5, 2), L-BFGS |
| Reproduction | Decision tree | Cost-complexity pruning, `ccp_alpha=0.003` |
| Reproduction | Gradient Boosting | Learning rate 0.01, multinomial classification |
| Extension | Random Forest | 300 trees, balanced subsampling, minimum leaf size 3 |

Each model uses the same date-sorted expanding-window `TimeSeriesSplit` folds and the same feature pipeline. Numeric imputation, categorical encoding, and scaling are fit separately inside each fold.

## Limitations

This is a close reproduction rather than a byte-for-byte replication. The original 3,355-row, 134-feature dataset is not available from the report's linked Drive folder. The replacement starts in 2010, and the source includes 118 overall fields before this project filters to pre-fight fighter data. Results can differ because of coverage, features, library versions, and underspecified training details in the reference.

The dataset has only five draws and three no-contests across its full date range, all outside the chosen 2010–2019 slice. Accuracy can hide differences between Red and Blue predictions, so the project also reports macro-F1. This work is for coursework and does not provide betting advice.

## Sources

- McQuaide, M. (2019), *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes*, Stanford CS229 Autumn 2019.
- Smith, B. (2026), UFC Athletes and Fight Data, TidyTuesday dataset documentation: <https://github.com/rfordatascience/tidytuesday/tree/main/data/2026/2026-07-07>
- Public source files are distributed in the TidyTuesday repository under its repository license. Dataset definitions are in `docs/data_notes.md`.
