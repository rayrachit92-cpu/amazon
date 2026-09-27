import sqlite3
import pandas as pd
from src.common import (
    fetch_candidates,
    norm_name,
    norm_address,
    norm_text,
)


SAMPLE_SIZE = 1000


def prepare_s1(row):
    return {
        "entity_id": row["entity_id"],
        "name_n": norm_name(row["business_name"]),
        "addr_n": norm_address(row["business_address"]),
        "country_n": norm_text(row["country"]),
    }


def load_ground_truth(path):
    gt = {}

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    for _, row in df.iterrows():
        ids = row["matched_entity_ids"]

        if not ids:
            gt[row["source1_entity_id"]] = set()
        else:
            gt[row["source1_entity_id"]] = set(
                x.strip() for x in ids.split(",") if x.strip()
            )

    return gt


def main():

    print("=" * 60)
    print("CANDIDATE RECALL TEST")
    print("=" * 60)

    print("\nLoading Source 1...")

    s1 = pd.read_csv(
        "dataset/train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Source 1 records:", len(s1))

    print("\nLoading ground truth...")

    gt = load_ground_truth(
        "dataset/train_ground_truth.tsv"
    )

    print("Ground truth entities:", len(gt))

    sample = s1.sample(
        n=min(SAMPLE_SIZE, len(s1)),
        random_state=42
    )

    print("\nSample size:", len(sample))

    print("\nOpening Source 2 database...")
    conn2 = sqlite3.connect(
        "work/train_source2.sqlite"
    )

    print("Opening Source 3 database...")
    conn3 = sqlite3.connect(
        "work/train_source3.sqlite"
    )

    total_true = 0
    found_true = 0

    entities_with_matches = 0
    entities_with_all_matches = 0

    total_candidates = 0

    missed_examples = []

    for i, (_, raw_row) in enumerate(sample.iterrows(), 1):

        row = prepare_s1(raw_row)

        true_ids = gt.get(
            row["entity_id"],
            set()
        )

        if true_ids:
            entities_with_matches += 1

        candidate_ids = set()

        for record in fetch_candidates(
            conn2,
            row,
            limit=500
        ):
            candidate_ids.add(record[0])

        for record in fetch_candidates(
            conn3,
            row,
            limit=500
        ):
            candidate_ids.add(record[0])

        total_candidates += len(candidate_ids)

        found = true_ids.intersection(candidate_ids)

        total_true += len(true_ids)
        found_true += len(found)

        if true_ids and found == true_ids:
            entities_with_all_matches += 1

        if true_ids and found != true_ids and len(missed_examples) < 20:

            missed = sorted(true_ids - candidate_ids)

            missed_examples.append({
                "entity_id": row["entity_id"],
                "name": raw_row["business_name"],
                "address": raw_row["business_address"],
                "country": raw_row["country"],
                "true_matches": len(true_ids),
                "found_matches": len(found),
                "missed_ids": ",".join(missed[:10])
            })

        if i % 100 == 0:
            print(
                f"Processed {i}/{len(sample)} | "
                f"true={total_true} | "
                f"found={found_true}"
            )

    conn2.close()
    conn3.close()

    recall = (
        found_true / total_true
        if total_true
        else 0
    )

    entity_recall = (
        entities_with_all_matches / entities_with_matches
        if entities_with_matches
        else 0
    )

    avg_candidates = (
        total_candidates / len(sample)
        if len(sample)
        else 0
    )

    print("\n")
    print("=" * 60)
    print("CANDIDATE RECALL RESULTS")
    print("=" * 60)

    print(f"\nS1 entities tested       : {len(sample):,}")
    print(f"Entities with matches    : {entities_with_matches:,}")
    print(f"True matches             : {total_true:,}")
    print(f"True matches found       : {found_true:,}")
    print(f"Missed true matches      : {total_true - found_true:,}")

    print(
        f"\nPAIR-LEVEL RECALL         : "
        f"{recall:.4%}"
    )

    print(
        f"ENTITY FULL-RECALL       : "
        f"{entity_recall:.4%}"
    )

    print(
        f"Average candidates/S1    : "
        f"{avg_candidates:.1f}"
    )

    print("\n")
    print("=" * 60)
    print("MISSED MATCH EXAMPLES")
    print("=" * 60)

    if not missed_examples:
        print("\nNo missed matches found.")
    else:
        for j, item in enumerate(missed_examples, 1):

            print(f"\n--- Example {j} ---")
            print("S1:", item["entity_id"])
            print("Name:", item["name"])
            print("Address:", item["address"])
            print("Country:", item["country"])
            print("True matches:", item["true_matches"])
            print("Found:", item["found_matches"])
            print("Missed IDs:", item["missed_ids"])


if __name__ == "__main__":
    main()