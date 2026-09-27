import argparse
from pathlib import Path

import pandas as pd


def inspect_source(path):

    print("\n==============================")
    print("SOURCE FILE:", path)
    print("==============================")

    rows = 0
    countries = {}

    missing_name = 0
    missing_address = 0
    missing_country = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000
    ):

        rows += len(chunk)

        missing_name += int(
            (chunk["business_name"].str.strip() == "").sum()
        )

        missing_address += int(
            (chunk["business_address"].str.strip() == "").sum()
        )

        missing_country += int(
            (chunk["country"].str.strip() == "").sum()
        )

        for country, count in chunk["country"].value_counts().items():
            countries[country] = (
                countries.get(country, 0) + int(count)
            )

    print("Rows:", f"{rows:,}")
    print("Missing business_name:", f"{missing_name:,}")
    print("Missing business_address:", f"{missing_address:,}")
    print("Missing country:", f"{missing_country:,}")

    print("\nCountries:")

    for country, count in sorted(
        countries.items(),
        key=lambda x: -x[1]
    ):
        print(f"{country}: {count:,}")


def inspect_ground_truth(path):

    print("\n==============================")
    print("GROUND TRUTH:", path)
    print("==============================")

    rows = 0
    empty = 0
    one_match = 0
    multi_match = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000
    ):

        rows += len(chunk)

        for value in chunk["matched_entity_ids"]:

            if not value.strip():
                empty += 1

            else:

                ids = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]

                if len(ids) == 1:
                    one_match += 1
                elif len(ids) > 1:
                    multi_match += 1

    print("Rows:", f"{rows:,}")
    print("Zero matches:", f"{empty:,}")
    print("Exactly one match:", f"{one_match:,}")
    print("Multiple matches:", f"{multi_match:,}")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="dataset"
)

args = parser.parse_args()

dataset = Path(args.dataset)

for filename in [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv"
]:

    inspect_source(
        dataset / filename
    )

inspect_ground_truth(
    dataset / "train_ground_truth.tsv"
)