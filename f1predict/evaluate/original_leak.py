"""Quantify the post-race leak in its native environment.

The modern pipeline rebuilds features from scratch, so running the original
hyperparameters against it does not isolate the leak -- the feature space
changed too. To measure what the leak was actually worth, this reproduces the
2023 setup exactly:

* the original committed ``data/Final.csv`` (2014-2022, 88 columns),
* the original target (``podium == 1`` binarised to "won the race"),
* the original split (train seasons <= 2021, test season 2022),
* the original scoring (per race, argmax of predicted win probability,
  ``precision_score`` against the actual winner),

and fits each model twice: once with the four ``status_*`` columns present, as
the original did, and once with them removed. The gap between the two is the
leak's contribution, measured rather than asserted.

    python -m f1predict.evaluate.original_leak
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ORIGINAL_DATASET = Path("data/Final.csv")
RESULTS_DIR = Path("results")

STATUS_COLUMNS = [
    "status_Finished",
    "status_Illness",
    "status_Incident",
    "status_Mechanical Issue",
]

#: Exactly what the original notebook dropped.
ORIGINAL_DROP = ["driver", "podium", "points"]


def build_models() -> dict:
    return {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=5000,
            random_state=1,
        ),
        "SVM": SVC(probability=True, gamma=0.1, C=10.0, kernel="sigmoid", random_state=1),
    }


def score_by_race(model, test: pd.DataFrame, feature_columns: list[str], scaler) -> float:
    """The original protocol: per race, does argmax(P(win)) land on the winner?"""
    hits, races = 0, 0
    for _, group in test.groupby("round"):
        X = scaler.transform(group[feature_columns])
        proba = model.predict_proba(X)
        scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        predicted = group.iloc[int(np.argmax(scores))]
        actual = group[group["podium"] == 1]
        if actual.empty:
            continue
        races += 1
        hits += int(predicted["driver"] == actual.iloc[0]["driver"])
    return hits / races if races else float("nan")


def run() -> dict:
    if not ORIGINAL_DATASET.exists():
        raise SystemExit(f"{ORIGINAL_DATASET} not found")

    df = pd.read_csv(ORIGINAL_DATASET)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    train = df[df["season"] <= 2021].copy()
    test = df[df["season"] == 2022].copy()

    y_train = (train["podium"] == 1).astype(int)

    results = {}
    for leak_present in (True, False):
        drop = list(ORIGINAL_DROP)
        if not leak_present:
            drop += [c for c in STATUS_COLUMNS if c in df.columns]
        feature_columns = [c for c in train.columns if c not in drop]

        scaler = StandardScaler().fit(train[feature_columns])
        X_train = scaler.transform(train[feature_columns])

        label = "with leak" if leak_present else "leak removed"
        for name, model in build_models().items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_train, y_train)
            accuracy = score_by_race(model, test, feature_columns, scaler)
            results[f"{name} ({label})"] = accuracy

    # The original's "second split": train on everything up to 2021 *plus*
    # rounds 1-11 of 2022, then score across all 22 rounds of 2022 -- which is
    # what its scoring function iterated over. Half the test races are training
    # rows. Reported separately because it is a different defect from the
    # status_* leak, and the audit suggests it is the one that produced 0.81.
    overlap_train = df[
        (df["season"] <= 2021) | ((df["season"] == 2022) & (df["round"] <= 11))
    ].copy()
    y_overlap = (overlap_train["podium"] == 1).astype(int)
    feature_columns = [c for c in df.columns if c not in ORIGINAL_DROP]
    scaler = StandardScaler().fit(overlap_train[feature_columns])
    X_overlap = scaler.transform(overlap_train[feature_columns])

    for name, model in build_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_overlap, y_overlap)
        # Scored over all 22 rounds, as the original did.
        results[f"{name} (overlapping split, all 22 rounds)"] = score_by_race(
            model, test, feature_columns, scaler
        )
        # And over only the genuinely held-out rounds, for contrast.
        results[f"{name} (overlapping split, rounds 12+ only)"] = score_by_race(
            model, test[test["round"] >= 12], feature_columns, scaler
        )

    # The naive rule to beat, on the same 2022 test season.
    #
    # `data/Final.csv` is raw Ergast, where a pit-lane start is grid == 0. Using
    # idxmin directly would nominate the pit-lane starter as the pole sitter --
    # 9 of the 22 races in 2022 have such an entry. That understates the
    # baseline as 0.364 when it is really 0.455, which is the difference between
    # the original SVM appearing to clear the bar and not clearing it.
    def pole_hits(group: pd.DataFrame) -> int:
        grid = pd.to_numeric(group["grid"], errors="coerce")
        starters = group[grid > 0]
        if starters.empty or not (group["podium"] == 1).any():
            return 0
        pole_sitter = starters.loc[
            pd.to_numeric(starters["grid"], errors="coerce").idxmin()
        ]["driver"]
        return int(pole_sitter == group[group["podium"] == 1].iloc[0]["driver"])

    pole = test.groupby("round").apply(pole_hits, include_groups=False)
    results["naive: pole sitter wins"] = float(pole.mean())
    results["_n_test_races"] = int(test["round"].nunique())
    results["_n_features_with_leak"] = int(
        len([c for c in train.columns if c not in ORIGINAL_DROP])
    )
    return results


def main() -> int:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "original_leak.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print()
    print("Original pipeline, original data (data/Final.csv), original split")
    print(f"train seasons <= 2021, test season 2022 "
          f"({results['_n_test_races']} races, "
          f"{results['_n_features_with_leak']} features)")
    print()
    print(f"{'configuration':32s} {'top-1':>8s}")
    print("-" * 42)
    for key, value in results.items():
        if key.startswith("_"):
            continue
        print(f"{key:32s} {value:8.3f}")

    leak_gaps = []
    for name in ("MLP", "SVM"):
        with_leak = results.get(f"{name} (with leak)")
        without = results.get(f"{name} (leak removed)")
        if with_leak is not None and without is not None:
            leak_gaps.append((name, with_leak - without))
    print()
    for name, gap in leak_gaps:
        print(f"  {name}: leak was worth {gap:+.3f} top-1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
