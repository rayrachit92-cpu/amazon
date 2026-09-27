import argparse
import json
import os
import random
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

from src.common import (
    FEATURE_NAMES,
    feature_row,
    fetch_candidates,
    load_ground_truth,
    macro_f05,
    tuple_to_record,
    norm_name,
    norm_address,
    norm_text,
)


def load_source1(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def get_record(conn, entity_id):
    row = conn.execute(
        "SELECT * FROM records WHERE entity_id = ?",
        (entity_id,)
    ).fetchone()

    if row is None:
        return None

    return tuple_to_record(row)


def prepare_s1_record(row):
    """
    Normalize Source-1 fields exactly the same
    way as records stored in SQLite.
    """

    return {
        "entity_id": row["entity_id"],
        "name_n": norm_name(
            row["business_name"]
        ),
        "addr_n": norm_address(
            row["business_address"]
        ),
        "country_n": norm_text(
            row["country"]
        ),
    }


def convert_candidate(candidate):

    if isinstance(candidate, dict):
        return candidate

    if hasattr(candidate, "keys"):
        return {
            key: candidate[key]
            for key in candidate.keys()
        }

    try:
        return tuple_to_record(candidate)
    except Exception:
        return None


def make_pairs(
    source1_df,
    gt,
    db2,
    db3,
    neg_per_source=15
):

    X = []
    y = []
    pair_ids = []

    total = len(source1_df)

    for pos, (_, row) in enumerate(
        source1_df.iterrows(),
        1
    ):

        s1 = prepare_s1_record(row)

        s1_id = s1["entity_id"]

        true_ids = set(
            gt.get(
                s1_id,
                set()
            )
        )

        positive_records = {}

        negative_records = {
            "S2": {},
            "S3": {}
        }

        # --------------------------------------------------
        # Directly retrieve TRUE matches
        # --------------------------------------------------

        for entity_id in true_ids:

            if entity_id.startswith("S2-"):

                rec = get_record(
                    db2,
                    entity_id
                )

                if rec is not None:
                    positive_records[
                        entity_id
                    ] = rec

            elif entity_id.startswith("S3-"):

                rec = get_record(
                    db3,
                    entity_id
                )

                if rec is not None:
                    positive_records[
                        entity_id
                    ] = rec

        # --------------------------------------------------
        # Generate hard negative candidates
        # --------------------------------------------------

        for source_name, conn in [
            ("S2", db2),
            ("S3", db3)
        ]:

            try:

                candidates = fetch_candidates(
                    conn,
                    s1,
                    limit=100
                )

            except Exception:
                candidates = []

            candidate_map = {}

            for candidate in candidates:

                rec = convert_candidate(
                    candidate
                )

                if rec is None:
                    continue

                cid = rec.get(
                    "entity_id"
                )

                if not cid:
                    continue

                candidate_map[
                    cid
                ] = rec

            negative_ids = [
                cid
                for cid in candidate_map
                if cid not in true_ids
            ]

            random.shuffle(
                negative_ids
            )

            for cid in negative_ids[
                :neg_per_source
            ]:

                negative_records[
                    source_name
                ][cid] = candidate_map[cid]

        # --------------------------------------------------
        # Add positive examples
        # --------------------------------------------------

        for entity_id, rec in (
            positive_records.items()
        ):

            try:

                features = feature_row(
                    s1,
                    rec
                )

            except Exception:
                continue

            X.append(features)
            y.append(1)

            pair_ids.append(
                (
                    s1_id,
                    entity_id
                )
            )

        # --------------------------------------------------
        # Add negative examples
        # --------------------------------------------------

        for source_name in [
            "S2",
            "S3"
        ]:

            for entity_id, rec in (
                negative_records[
                    source_name
                ].items()
            ):

                try:

                    features = feature_row(
                        s1,
                        rec
                    )

                except Exception:
                    continue

                X.append(features)
                y.append(0)

                pair_ids.append(
                    (
                        s1_id,
                        entity_id
                    )
                )

        if (
            pos % 50 == 0
            or pos == total
        ):

            print(
                f"Processed {pos}/{total} "
                f"S1 entities | "
                f"pairs={len(X)} | "
                f"positives={sum(y)} | "
                f"negatives={len(y) - sum(y)}"
            )

    if not X:
        raise RuntimeError(
            "No training pairs were generated."
        )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.int8
        ),
        pair_ids
    )


def evaluate(
    source1_df,
    gt,
    model,
    db2,
    db3,
    threshold=0.50
):

    print(
        "\nRunning validation..."
    )

    predictions = {}

    total = len(
        source1_df
    )

    for pos, (_, row) in enumerate(
        source1_df.iterrows(),
        1
    ):

        s1 = prepare_s1_record(
            row
        )

        s1_id = s1["entity_id"]

        all_candidates = {}

        # --------------------------------------------------
        # Get candidates from S2 and S3
        # --------------------------------------------------

        for source_name, conn in [
            ("S2", db2),
            ("S3", db3)
        ]:

            try:

                candidates = fetch_candidates(
                    conn,
                    s1,
                    limit=100
                )

            except Exception:
                candidates = []

            for candidate in candidates:

                rec = convert_candidate(
                    candidate
                )

                if rec is None:
                    continue

                cid = rec.get(
                    "entity_id"
                )

                if cid:

                    all_candidates[
                        cid
                    ] = rec

        # --------------------------------------------------
        # Score candidates
        # --------------------------------------------------

        if all_candidates:

            ids = list(
                all_candidates.keys()
            )

            records = [
                all_candidates[cid]
                for cid in ids
            ]

            features = []

            for rec in records:

                try:

                    features.append(
                        feature_row(
                            s1,
                            rec
                        )
                    )

                except Exception:

                    features.append(
                        [0.0]
                        * len(FEATURE_NAMES)
                    )

            X = np.asarray(
                features,
                dtype=np.float32
            )

            scores = model.predict_proba(
                X
            )[:, 1]

            predictions[
                s1_id
            ] = {
                cid
                for cid, score in zip(
                    ids,
                    scores
                )
                if score >= threshold
            }

        else:

            predictions[
                s1_id
            ] = set()

        if (
            pos % 25 == 0
            or pos == total
        ):

            print(
                f"Validation "
                f"{pos}/{total}"
            )

    score = macro_f05(
        predictions,
        gt
    )

    return score


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500
    )

    parser.add_argument(
        "--neg-per-source",
        type=int,
        default=15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    print("=" * 60)

    print(
        "FAST BUSINESS ENTITY RESOLUTION TRAINING"
    )

    print("=" * 60)

    # ------------------------------------------------------
    # Load Source 1
    # ------------------------------------------------------

    print(
        "\nLoading Source 1..."
    )

    source1 = load_source1(
        "dataset/train_source1.tsv"
    )

    print(
        f"Source 1 records: "
        f"{len(source1):,}"
    )

    # ------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    gt = load_ground_truth(
        "dataset/train_ground_truth.tsv"
    )

    print(
        f"Ground truth entities: "
        f"{len(gt):,}"
    )

    # ------------------------------------------------------
    # Sample S1 entities
    # ------------------------------------------------------

    sample_size = min(
        args.sample_size,
        len(source1)
    )

    sample = source1.sample(
        n=sample_size,
        random_state=args.seed
    ).reset_index(
        drop=True
    )

    split = int(
        len(sample) * 0.8
    )

    train_df = sample.iloc[
        :split
    ].copy()

    valid_df = sample.iloc[
        split:
    ].copy()

    print(
        f"\nTraining S1 entities: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation S1 entities: "
        f"{len(valid_df):,}"
    )

    # ------------------------------------------------------
    # Open SQLite databases
    # ------------------------------------------------------

    print(
        "\nOpening Source 2 database..."
    )

    db2 = sqlite3.connect(
        "work/train_source2.sqlite"
    )

    print(
        "Opening Source 3 database..."
    )

    db3 = sqlite3.connect(
        "work/train_source3.sqlite"
    )

    # ------------------------------------------------------
    # Generate training data
    # ------------------------------------------------------

    print(
        "\nGenerating training pairs..."
    )

    X_train, y_train, _ = make_pairs(
        train_df,
        gt,
        db2,
        db3,
        neg_per_source=args.neg_per_source
    )

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    print(
        "\nTraining matrix:"
    )

    print(
        "Rows:",
        len(X_train)
    )

    print(
        "Features:",
        X_train.shape[1]
    )

    print(
        "Positive:",
        positive_count
    )

    print(
        "Negative:",
        negative_count
    )

    # ------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------

    if positive_count == 0:

        raise RuntimeError(
            "No positive training examples."
        )

    if negative_count == 0:

        raise RuntimeError(
            "No negative training examples."
        )

    scale_pos_weight = min(
        negative_count /
        positive_count,
        20.0
    )

    print(
        "scale_pos_weight:",
        round(
            scale_pos_weight,
            3
        )
    )

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    print(
        "\nTraining XGBoost..."
    )

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nModel training completed."
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    score = evaluate(
        valid_df,
        gt,
        model,
        db2,
        db3,
        threshold=0.50
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"Validation Macro F0.5: "
        f"{score:.6f}"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------
    # Save model
    # ------------------------------------------------------

    os.makedirs(
        "work",
        exist_ok=True
    )

    model_path = (
        "work/entity_matcher_fast.json"
    )

    config_path = (
        "work/model_config_fast.json"
    )

    model.save_model(
        model_path
    )

    config = {
        "threshold": 0.50,
        "features": FEATURE_NAMES,
        "sample_size": args.sample_size,
        "neg_per_source": args.neg_per_source,
        "validation_f05": float(score)
    }

    with open(
        config_path,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    print(
        "\nSaved:"
    )

    print(
        model_path
    )

    print(
        config_path
    )

    db2.close()
    db3.close()


if __name__ == "__main__":
    main()