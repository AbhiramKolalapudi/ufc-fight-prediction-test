from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
PLOTS = ROOT / "results" / "plots"
SUMMARY = pd.read_csv(ROOT / "results" / "model_comparison.csv")
FOLDS = pd.read_csv(ROOT / "results" / "fold_metrics.csv")
PAPER_ACCURACY = {
    "SGD perceptron": 0.58018,
    "MLP (5, 2)": 0.57736,
    "Decision tree": 0.60252,
    "Gradient boosting": 0.61226,
}
EXPLANATIONS = {
    "SGD perceptron": {
        "file": "01_sgd_perceptron.ipynb",
        "title": "SGD Perceptron",
        "idea": "A linear classifier learns one weighted score from the fighter statistics. Stochastic Gradient Descent updates those weights after examples, while perceptron loss focuses on correcting mistakes. The predicted corner is the side with the larger score.",
        "why": "This is the paper's linear baseline. It trains quickly and is easy to interpret at a high level, but a single linear boundary cannot capture complex interactions between fighter attributes.",
        "settings": "Perceptron loss; random seed 42; default regularization and learning-rate settings from scikit-learn.",
        "reading": "A small train-test gap is consistent with a simple boundary. If results are near the always-Red baseline, the model may not be extracting much signal beyond the class imbalance.",
    },
    "MLP (5, 2)": {
        "file": "02_mlp_neural_network.ipynb",
        "title": "Multilayer Perceptron",
        "idea": "A small neural network passes the fighter features through two hidden layers with 5 and 2 neurons. Nonlinear activation functions let it combine attributes in ways a linear model cannot. L-BFGS optimizes its weights, with log-loss as the objective.",
        "why": "The paper tests whether even a compact neural network can find nonlinear patterns in the pre-fight statistics.",
        "settings": "Hidden layers (5, 2); L-BFGS solver; log-loss objective; random seed 42; maximum 200 iterations.",
        "reading": "Some folds hit the 200-iteration limit, so the saved result is not guaranteed to be fully converged. A high training score with a lower later-period test score would indicate overfitting.",
    },
    "Decision tree": {
        "file": "03_decision_tree.ipynb",
        "title": "Pruned Decision Tree",
        "idea": "A decision tree repeatedly asks questions such as whether one feature is above a threshold, then assigns a corner at the end of each path. Cost-complexity pruning penalizes unnecessary branches; ccp_alpha controls that penalty.",
        "why": "Trees can express nonlinear rules and feature interactions while producing a structure that can be inspected. Pruning helps limit complexity.",
        "settings": "Cost-complexity pruning with ccp_alpha=0.003; random seed 42.",
        "reading": "Compare training and test accuracy. A large gap suggests that the tree's rules fit historical training bouts too closely to generalize to later bouts.",
    },
    "Gradient boosting": {
        "file": "04_gradient_boosting.ipynb",
        "title": "Gradient Boosting",
        "idea": "Gradient Boosting builds a sequence of small decision trees. Each new tree focuses on errors left by the current ensemble, and the learning rate controls how strongly each tree contributes.",
        "why": "The paper reports this as its strongest model. Boosted trees can capture nonlinear relationships while adding complexity gradually.",
        "settings": "Learning rate 0.01; scikit-learn GradientBoostingClassifier defaults otherwise; random seed 42.",
        "reading": "This model achieved the best mean accuracy among the reproduced models in this dataset slice, but that score matches the always-Red baseline. Compare macro-F1 as well as accuracy.",
    },
    "Random forest (extension)": {
        "file": "05_random_forest_extension.ipynb",
        "title": "Random Forest Extension",
        "idea": "A Random Forest trains many decision trees on bootstrapped samples and random subsets of features, then combines their votes. Averaging many diverse trees usually reduces the instability of one deep tree.",
        "why": "Random Forest is not one of the reference models. It provides a tree-ensemble comparison that captures nonlinear effects and interactions without building trees sequentially.",
        "settings": "300 trees; minimum leaf size 3; balanced subsampling by class; random seed 42.",
        "reading": "The extension ties Gradient Boosting on mean test accuracy and has higher mean macro-F1, but its very high training accuracy shows substantial overfitting. It does not improve accuracy over the paper reproduction.",
    },
}


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def result_markdown(name: str, details: dict) -> str:
    result = SUMMARY.loc[SUMMARY["model"] == name].iloc[0]
    paper = PAPER_ACCURACY.get(name)
    paper_text = f"{paper:.1%}" if paper is not None else "Not reported in the paper"
    delta = f"{result['test_accuracy'] - paper:+.1%}" if paper is not None else "—"
    model_folds = FOLDS.loc[FOLDS["model"] == name]
    table_rows = "\n".join(
        f"| {int(row.fold)} | {row.train_accuracy:.1%} | {row.test_accuracy:.1%} | {row.test_macro_f1:.3f} |"
        for row in model_folds.itertuples()
    )
    plot_path = f"../results/plots/{Path(details['file']).stem}_accuracy.png"
    return f"""## Saved run results

These are the results from the completed 20-fold chronological run. The next cells rerun this model from the bundled dataset. Accuracy is the share of bouts assigned to the correct corner. Macro-F1 computes F1 separately for Red and Blue, then averages the two scores equally.

| Measure | Result |
|---|---:|
| Mean training accuracy | {result['train_accuracy']:.1%} |
| Mean test accuracy | {result['test_accuracy']:.1%} |
| Test accuracy standard deviation across folds | {result['test_accuracy_sd']:.1%} |
| Mean test macro-F1 | {result['test_macro_f1']:.3f} |
| Paper test accuracy | {paper_text} |
| Difference from paper | {delta} |

The always-Red baseline is 58.2% accuracy on this slice. This period contains 4,073 bouts: 2,372 Red wins and 1,701 Blue wins. It contains no draws, so results here are binary despite the reference paper's three-outcome task.

![Twenty chronological test-fold accuracies](""" + plot_path + f""")

### Fold-by-fold results

| Fold | Train accuracy | Test accuracy | Test macro-F1 |
|---:|---:|---:|---:|
{table_rows}

**How to read this model's result:** {details['reading']}
"""


def make_plot(name: str, details: dict) -> None:
    model_folds = FOLDS.loc[FOLDS["model"] == name]
    figure, axis = plt.subplots(figsize=(9, 3.8))
    axis.plot(model_folds["fold"], model_folds["test_accuracy"], marker="o", color="#ef6c3b", label="Model test accuracy")
    axis.axhline(0.582, color="#338cc7", linestyle="--", linewidth=1.7, label="Always-Red baseline (58.2%)")
    if name in PAPER_ACCURACY:
        axis.axhline(PAPER_ACCURACY[name], color="#49a887", linestyle=":", linewidth=1.7, label="Paper accuracy")
    axis.set_title("Test accuracy by chronological fold")
    axis.set_xlabel("Fold, ordered by fight date")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.25, 1.0)
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOTS / f"{Path(details['file']).stem}_accuracy.png", dpi=150, facecolor="white")
    plt.close(figure)


def make_notebook(name: str, details: dict) -> dict:
    notebook_cells = [
        markdown_cell(f"""# {details['title']}: UFC fight outcomes

## Question

Can pre-fight fighter records predict whether the Red or Blue corner wins? The reference study is McQuaide (2019), *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes*. Its original dataset link is unavailable, so this project uses the public TidyTuesday UFC dataset and restricts it to 2010–2019.

## How this model works

{details['idea']}

### Why include it

{details['why']}

### Settings used

{details['settings']}

## Shared evaluation design

Each notebook uses the same 48 pre-fight fighter fields, date ordering, 20 expanding-window time-series folds, and preprocessing as the main project. Each test fold occurs after its training data. Numeric gaps are filled with the training-fold median, stances are one-hot encoded, and features are scaled inside the fold. No-contests are excluded. The selected years have no draws.

The saved tables below let you inspect the completed run immediately. Run the code cells to reproduce the results on your machine.
"""),
        markdown_cell(result_markdown(name, details)),
        code_cell(f"""from pathlib import Path
import sys
from IPython.display import display
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

search_paths = [Path.cwd(), *Path.cwd().parents]
project_root = next((path for path in search_paths if (path / "src" / "train.py").is_file()), None)
if project_root is None:
    raise FileNotFoundError("Open the UFC project folder in VS Code so the notebook can locate src/train.py.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.train import N_SPLITS, load_data, make_models, make_pipeline

data, X, y = load_data()
model_name = {name!r}
model = make_models()[model_name]
print(f"Loaded {{len(data):,}} bouts from {{data['date'].min().date()}} to {{data['date'].max().date()}}")
print("Outcome counts:")
display(y.value_counts().rename_axis("corner").to_frame("bouts"))"""),
        code_cell("""fold_rows = []
splitter = TimeSeriesSplit(n_splits=N_SPLITS)

for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)
    train_prediction = pipeline.predict(X_train)
    test_prediction = pipeline.predict(X_test)
    fold_rows.append({
        "fold": fold,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_accuracy": accuracy_score(y_train, train_prediction),
        "test_accuracy": accuracy_score(y_test, test_prediction),
        "test_macro_f1": f1_score(y_test, test_prediction, average="macro", zero_division=0),
    })

fold_results = pd.DataFrame(fold_rows)
summary = pd.DataFrame([{
    "model": model_name,
    "mean_train_accuracy": fold_results["train_accuracy"].mean(),
    "mean_test_accuracy": fold_results["test_accuracy"].mean(),
    "test_accuracy_sd": fold_results["test_accuracy"].std(),
    "mean_test_macro_f1": fold_results["test_macro_f1"].mean(),
}])
display(summary.style.format({
    "mean_train_accuracy": "{:.1%}",
    "mean_test_accuracy": "{:.1%}",
    "test_accuracy_sd": "{:.1%}",
    "mean_test_macro_f1": "{:.3f}",
}))
display(fold_results.style.format({
    "train_accuracy": "{:.1%}",
    "test_accuracy": "{:.1%}",
    "test_macro_f1": "{:.3f}",
}))"""),
        code_cell("""import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(fold_results["fold"], fold_results["test_accuracy"], marker="o", color="#ef6c3b", label="Model test accuracy")
ax.axhline(0.582, color="#338cc7", linestyle="--", label="Always-Red baseline (58.2%)")
ax.set(title="Test accuracy by chronological fold", xlabel="Fold, ordered by fight date", ylabel="Accuracy")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()"""),
        markdown_cell(f"""## Interpretation

{details['reading']}

The reference scores are not a like-for-like benchmark because its original sample spans 1997–2019, has 134 features, and includes three outcomes. This dataset starts in 2010 and the selected slice has only Red and Blue outcomes. Treat the paper comparison as approximate.

**Sources:** [McQuaide (2019) report](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647731.pdf) · [TidyTuesday dataset documentation](https://github.com/rfordatascience/tidytuesday/tree/main/data/2026/2026-07-07)
"""),
    ]
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name, details in EXPLANATIONS.items():
        make_plot(name, details)
        path = NOTEBOOKS / details["file"]
        path.write_text(json.dumps(make_notebook(name, details), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
