import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.common import (
    build_db,
    feature_row,
    fetch_candidates,
    load_ground_truth,
    macro_f05,
    prepare_chunk,
    read_source,
    tuple_to_record,
    FEATURE_NAMES
)


def load_source1(path):

    chunks = []

    for chunk in read_source(path):

        chunks.append(
            prepare_chunk(chunk)
        )

    return pd.concat(
        chunks,
        ignore_index=True
    )


def create_pairs(
    source1,
    databases,
    ground_truth,
    selected_ids
):

    X = []
    y = []
    pair_ids = []

    connections = []

    import sqlite3

    for database in databases:
        connections.append(
            sqlite3.connect(database)
        )

    for counter, (_, row) in enumerate(
        source1.iterrows()
    ):

        entity_id = row["entity_id"]

        if entity_id not in selected_ids:
            continue

        if counter % 1000 == 0:
            print(
                "Processed Source-1:",
                counter
            )

        source1_record = row.to_dict()

        actual = ground_truth.get(
            entity_id,
            set()
        )

        seen = set()

        for connection in connections:

            for candidate_tuple in fetch_candidates(
                connection,
                source1_record,
                limit=500
            ):

                candidate_id = candidate_tuple[0]

                if candidate_id in seen:
                    continue

                seen.add(candidate_id)

                candidate = tuple_to_record(
                    candidate_tuple
                )

                X.append(
                    feature_row(
                        source1_record,
                        candidate
                    )
                )

                y.append(
                    1
                    if candidate_id in actual
                    else 0
                )

                pair_ids.append(
                    (
                        entity_id,
                        candidate_id
                    )
                )

        # Ensure known positive pairs are represented.
        for positive_id in actual:

            if positive_id in seen:
                continue

            for connection in connections:

                candidate_tuple = connection.execute(
                    """
                    SELECT *
                    FROM records
                    WHERE entity_id = ?
                    """,
                    (positive_id,)
                ).fetchone()

                if candidate_tuple:

                    candidate = tuple_to_record(
                        candidate_tuple
                    )

                    X.append(
                        feature_row(
                            source1_record,
                            candidate
                        )
                    )

                    y.append(1)

                    pair_ids.append(
                        (
                            entity_id,
                            positive_id
                        )
                    )

                    break

    for connection in connections:
        connection.close()

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
        pair_ids
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="dataset"
    )

    parser.add_argument(
        "--work",
        default="work"
    )

    parser.add_argument(
        "--output",
        default="output"
    )

    args = parser.parse_args()

    dataset = Path(args.dataset)
    work = Path(args.work)
    output = Path(args.output)

    work.mkdir(
        parents=True,
        exist_ok=True
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== BUILD INDEXES ==========\n")

    build_db(
        dataset / "train_source2.tsv",
        work / "train_source2.sqlite"
    )

    build_db(
        dataset / "train_source3.tsv",
        work / "train_source3.sqlite"
    )

    print("\n========== LOAD SOURCE 1 ==========\n")

    source1 = load_source1(
        dataset / "train_source1.tsv"
    )

    ground_truth = load_ground_truth(
        dataset / "train_ground_truth.tsv"
    )

    ids = source1["entity_id"].tolist()

    rng = np.random.default_rng(42)

    rng.shuffle(ids)

    split = int(
        len(ids) * 0.80
    )

    train_ids = set(
        ids[:split]
    )

    validation_ids = set(
        ids[split:]
    )

    print(
        "Total Source-1:",
        len(ids)
    )

    print(
        "Training:",
        len(train_ids)
    )

    print(
        "Validation:",
        len(validation_ids)
    )

    databases = [
        work / "train_source2.sqlite",
        work / "train_source3.sqlite"
    ]

    print("\n========== CREATE TRAINING PAIRS ==========\n")

    X_train, y_train, _ = create_pairs(
        source1,
        databases,
        ground_truth,
        train_ids
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Positive pairs:",
        int(y_train.sum())
    )

    print(
        "Negative pairs:",
        int((y_train == 0).sum())
    )

    print("\n========== CREATE VALIDATION PAIRS ==========\n")

    X_valid, y_valid, valid_pairs = create_pairs(
        source1,
        databases,
        ground_truth,
        validation_ids
    )

    print(
        "Validation shape:",
        X_valid.shape
    )

    print("\n========== TRAIN MODEL ==========\n")

    positives = max(
        1,
        int(y_train.sum())
    )

    negatives = max(
        1,
        int((y_train == 0).sum())
    )

    model = XGBClassifier(

        n_estimators=500,

        max_depth=7,

        learning_rate=0.06,

        subsample=0.85,

        colsample_bytree=0.85,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        n_jobs=max(
            1,
            (os.cpu_count() or 2) - 1
        ),

        random_state=42,

        scale_pos_weight=min(
            negatives / positives,
            50
        )
    )

    model.fit(
        X_train,
        y_train
    )

    print("\n========== THRESHOLD SEARCH ==========\n")

    probabilities = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    validation_truth = {
        entity_id: ground_truth[entity_id]
        for entity_id in validation_ids
        if entity_id in ground_truth
    }

    best_threshold = 0.90
    best_score = -1

    for threshold in np.arange(
        0.50,
        1.00,
        0.01
    ):

        predictions = {}

        for pair, probability in zip(
            valid_pairs,
            probabilities
        ):

            source1_id, candidate_id = pair

            if probability >= threshold:

                predictions.setdefault(
                    source1_id,
                    set()
                ).add(
                    candidate_id
                )

        score = macro_f05(
            predictions,
            validation_truth
        )

        print(
            f"threshold={threshold:.2f} "
            f"F0.5={score:.6f}"
        )

        if score > best_score:

            best_score = score
            best_threshold = float(
                threshold
            )

    print("\n===================================")
    print(
        f"BEST THRESHOLD: {best_threshold:.2f}"
    )
    print(
        f"VALIDATION F0.5: {best_score:.6f}"
    )
    print("===================================\n")

    model.save_model(
        work / "entity_matcher.json"
    )

    config = {

        "threshold":
            best_threshold,

        "validation_macro_f05":
            best_score,

        "features":
            FEATURE_NAMES,

        "model":
            "XGBoost"

    }

    with open(
        work / "model_config.json",
        "w"
    ) as file:

        json.dump(
            config,
            file,
            indent=2
        )

    print(
        "Model saved."
    )


if __name__ == "__main__":
    main()
    