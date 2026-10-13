from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "ultimate_ufc_dataset.csv"
RESULTS = ROOT / "results"
SEED = 42
N_SPLITS = 20
START_DATE = "2010-01-01"
END_DATE = "2019-12-31"

FIGHTER_NUMERIC = [
    "current_lose_streak", "current_win_streak", "draw", "avg_sig_str_landed",
    "avg_sig_str_pct", "avg_sub_att", "avg_td_landed", "avg_td_pct",
    "longest_win_streak", "losses", "total_rounds_fought", "total_title_bouts",
    "win_by_decision_majority", "win_by_decision_split", "win_by_decision_unanimous",
    "win_by_ko_tko", "win_by_submission", "win_by_tko_doctor_stoppage", "wins",
    "height_cms", "reach_cms", "weight_lbs", "age",
]
NUMERIC_FEATURES = [f"{corner}_{feature}" for corner in ("r", "b") for feature in FIGHTER_NUMERIC]
CATEGORICAL_FEATURES = ["r_stance", "b_stance"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_models() -> dict[str, object]:
    """Paper models plus a Random Forest extension, using reported core settings."""
    return {
        "SGD perceptron": SGDClassifier(loss="perceptron", random_state=SEED),
        "MLP (5, 2)": MLPClassifier(
            hidden_layer_sizes=(5, 2), solver="lbfgs", random_state=SEED,
            max_iter=200, tol=1e-4,
        ),
        "Decision tree": DecisionTreeClassifier(ccp_alpha=0.003, random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(learning_rate=0.01, random_state=SEED),
        "Random forest (extension)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1,
        ),
    }


def make_pipeline(model: object) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("preprocess", preprocess),
        ("scale", StandardScaler()),
        ("model", clone(model)),
    ])


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"].between(START_DATE, END_DATE)].copy()
    data = data[data["winner"].isin(["Red", "Blue", "Draw"])].copy()
    data = data.sort_values("date", kind="stable").reset_index(drop=True)
    features = data[FEATURES].copy()
    labels = data["winner"].astype(str)
    return data, features, labels


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data, X, y = load_data()
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    models = make_models()
    fold_rows: list[dict[str, float | int | str]] = []
    last_fold_examples: list[dict[str, str | float]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        for name, model in models.items():
            pipeline = make_pipeline(model)
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            fold_rows.append({
                "model": name,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_accuracy": accuracy_score(y_train, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
            })
            if fold == N_SPLITS and name == "Gradient boosting":
                for idx, predicted in zip(test_idx, test_pred):
                    last_fold_examples.append({
                        "date": data.iloc[idx]["date"].strftime("%Y-%m-%d"),
                        "red_fighter": str(data.iloc[idx]["r_fighter"]),
                        "blue_fighter": str(data.iloc[idx]["b_fighter"]),
                        "actual": str(y.iloc[idx]),
                        "predicted": str(predicted),
                    })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(RESULTS / "fold_metrics.csv", index=False)
    summary = (fold_df.groupby("model", sort=False)
               .agg(train_accuracy=("train_accuracy", "mean"),
                    test_accuracy=("test_accuracy", "mean"),
                    test_accuracy_sd=("test_accuracy", "std"),
                    test_macro_f1=("test_macro_f1", "mean"))
               .reset_index())
    summary.to_csv(RESULTS / "model_comparison.csv", index=False)
    if last_fold_examples:
        pd.DataFrame(last_fold_examples).to_csv(RESULTS / "demo_predictions.csv", index=False)

    with (RESULTS / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_file": str(DATA_PATH.relative_to(ROOT)),
            "source_rows_after_period_and_label_filter": len(data),
            "period": [START_DATE, END_DATE],
            "label_counts": y.value_counts().to_dict(),
            "feature_count_before_encoding": len(FEATURES),
            "numeric_feature_count": len(NUMERIC_FEATURES),
            "categorical_features": CATEGORICAL_FEATURES,
            "validation": f"TimeSeriesSplit(n_splits={N_SPLITS}), date-sorted, expanding training windows",
            "random_seed": SEED,
            "excluded_labels": ["No Contest"],
        }, f, indent=2)

    print(f"Rows: {len(data):,} | Date range: {data.date.min().date()} to {data.date.max().date()}")
    print("Class counts:", y.value_counts().to_dict())
    print(summary.to_string(index=False, formatters={
        "train_accuracy": "{:.3f}".format,
        "test_accuracy": "{:.3f}".format,
        "test_accuracy_sd": "{:.3f}".format,
        "test_macro_f1": "{:.3f}".format,
    }))
    print(f"Saved results to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
