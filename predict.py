import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.common import (
    build_db,
    feature_row,
    fetch_candidates,
    prepare_chunk,
    read_source,
    tuple_to_record
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

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        work / "model_config.json"
    ) as file:

        config = json.load(file)

    threshold = float(
        config["threshold"]
    )

    print(
        "Using threshold:",
        threshold
    )

    model = XGBClassifier()

    model.load_model(
        work / "entity_matcher.json"
    )

    print("\n========== BUILD TEST INDEXES ==========\n")

    build_db(
        dataset / "test" / "test_source2.tsv",
        work / "test_source2.sqlite"
    )

    build_db(
        dataset / "test" / "test_source3.tsv",
        work / "test_source3.sqlite"
    )

    conn2 = sqlite3.connect(
        work / "test_source2.sqlite"
    )

    conn3 = sqlite3.connect(
        work / "test_source3.sqlite"
    )

    connections = [
        conn2,
        conn3
    ]

    matching_file = (
        output /
        "matching_results.tsv"
    )

    candidate_file = (
        output /
        "candidate_pairs.tsv"
    )

    with open(
        matching_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as matching:

        with open(
            candidate_file,
            "w",
            encoding="utf-8",
            newline=""
        ) as candidates_file:

            matching_writer = csv.writer(
                matching,
                delimiter="\t",
                lineterminator="\n"
            )

            candidate_writer = csv.writer(
                candidates_file,
                delimiter="\t",
                lineterminator="\n"
            )

            matching_writer.writerow(
                [
                    "source1_entity_id",
                    "matched_entity_ids"
                ]
            )

            candidate_writer.writerow(
                [
                    "source1_entity_id",
                    "candidate_entity_ids"
                ]
            )

            processed = 0

            for chunk in read_source(
                dataset /
                "test" /
                "test_source1.tsv"
            ):

                chunk = prepare_chunk(
                    chunk
                )

                for _, row in chunk.iterrows():

                    processed += 1

                    if processed % 1000 == 0:

                        print(
                            "Processed test entities:",
                            processed
                        )

                    source1 = row.to_dict()

                    candidates = []

                    seen = set()

                    for connection in connections:

                        for record in fetch_candidates(
                            connection,
                            source1,
                            limit=500
                        ):

                            candidate_id = record[0]

                            if candidate_id in seen:
                                continue

                            seen.add(
                                candidate_id
                            )

                            candidates.append(
                                tuple_to_record(
                                    record
                                )
                            )

                    candidate_ids = [
                        candidate["entity_id"]
                        for candidate in candidates
                    ]

                    matches = []

                    if candidates:

                        X = np.asarray(
                            [
                                feature_row(
                                    source1,
                                    candidate
                                )
                                for candidate in candidates
                            ],
                            dtype=np.float32
                        )

                        probabilities = (
                            model.predict_proba(X)
                            [:, 1]
                        )

                        for candidate, probability in zip(
                            candidates,
                            probabilities
                        ):

                            if float(
                                probability
                            ) >= threshold:

                                matches.append(
                                    candidate[
                                        "entity_id"
                                    ]
                                )

                    matches = list(
                        dict.fromkeys(
                            matches
                        )
                    )

                    candidate_ids = list(
                        dict.fromkeys(
                            candidate_ids
                        )
                    )

                    matching_writer.writerow(
                        [
                            source1["entity_id"],
                            ",".join(matches)
                        ]
                    )

                    candidate_writer.writerow(
                        [
                            source1["entity_id"],
                            ",".join(candidate_ids)
                        ]
                    )

    conn2.close()
    conn3.close()

    print("\n========== COMPLETE ==========\n")

    print(
        "Matching file:",
        matching_file
    )

    print(
        "Candidate file:",
        candidate_file
    )


if __name__ == "__main__":
    main()
    