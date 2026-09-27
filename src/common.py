import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


SOURCE_COLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]


LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "llc",
    "plc", "pvt", "private", "llp",
    "sarl", "sa", "sas", "gmbh", "ag", "bv"
}


ADDRESS_MAP = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "av": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "square": "sq",
    "sq": "sq",
    "building": "bldg",
    "bldg": "bldg",
    "floor": "fl",
    "fl": "fl",
    "apartment": "apt",
    "apt": "apt",
    "suite": "ste",
    "ste": "ste"
}


def norm_text(value):
    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()

    value = value.replace(
        "&",
        " and "
    )

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def norm_name(value):
    text = norm_text(value)

    tokens = [
        token
        for token in text.split()
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


def norm_address(value):
    text = norm_text(value)

    output = []

    for token in text.split():
        output.append(
            ADDRESS_MAP.get(
                token,
                token
            )
        )

    return " ".join(output)


def prepare_chunk(df):

    df = df.copy()

    df["name_n"] = (
        df["business_name"]
        .map(norm_name)
    )

    df["addr_n"] = (
        df["business_address"]
        .map(norm_address)
    )

    df["country_n"] = (
        df["country"]
        .map(norm_text)
    )

    return df


def read_source(
    path,
    chunksize=100_000
):

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=SOURCE_COLS,
        chunksize=chunksize
    )


def build_db(
    tsv_path,
    db_path
):

    db_path = Path(db_path)

    if db_path.exists():
        print(
            "Database already exists:",
            db_path
        )
        return

    print(
        "Creating:",
        db_path
    )

    conn = sqlite3.connect(
        db_path
    )

    conn.execute(
        """
        PRAGMA journal_mode=WAL
        """
    )

    conn.execute(
        """
        PRAGMA synchronous=NORMAL
        """
    )

    conn.execute(
        """
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            name_n TEXT,
            addr_n TEXT,
            country_n TEXT
        )
        """
    )

    for chunk in read_source(
        tsv_path
    ):

        chunk = prepare_chunk(
            chunk
        )

        rows = chunk[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country",
                "name_n",
                "addr_n",
                "country_n"
            ]
        ].itertuples(
            index=False,
            name=None
        )

        conn.executemany(
            """
            INSERT OR REPLACE INTO records
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )

        conn.commit()

    conn.execute(
        """
        CREATE INDEX idx_name_country
        ON records(name_n, country_n)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_addr_country
        ON records(addr_n, country_n)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_name_prefix
        ON records(
            substr(name_n, 1, 4),
            country_n
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_addr_prefix
        ON records(
            substr(addr_n, 1, 8),
            country_n
        )
        """
    )

    conn.commit()
    conn.close()

    print(
        "Finished:",
        db_path
    )


# ==========================================================
# CANDIDATE GENERATION
# ==========================================================

def fetch_candidates(
    conn,
    row,
    limit=500
):

    seen = set()

    name = row.get(
        "name_n",
        ""
    )

    addr = row.get(
        "addr_n",
        ""
    )

    country = row.get(
        "country_n",
        ""
    )

    queries = []

    # ------------------------------------------------------
    # 1. Exact normalized name
    # ------------------------------------------------------

    if name:

        queries.append(
            (
                """
                SELECT *
                FROM records
                WHERE name_n = ?
                AND country_n = ?
                LIMIT ?
                """,
                (
                    name,
                    country,
                    100
                )
            )
        )

    # ------------------------------------------------------
    # 2. Exact normalized address
    # ------------------------------------------------------

    if addr:

        queries.append(
            (
                """
                SELECT *
                FROM records
                WHERE addr_n = ?
                AND country_n = ?
                LIMIT ?
                """,
                (
                    addr,
                    country,
                    100
                )
            )
        )

    # ------------------------------------------------------
    # 3. Name prefix blocks
    # ------------------------------------------------------

    for prefix_len in (
        4,
        6,
        8
    ):

        if len(name) >= prefix_len:

            queries.append(
                (
                    """
                    SELECT *
                    FROM records
                    WHERE substr(
                        name_n,
                        1,
                        ?
                    ) = ?
                    AND country_n = ?
                    LIMIT ?
                    """,
                    (
                        prefix_len,
                        name[:prefix_len],
                        country,
                        100
                    )
                )
            )

    # ------------------------------------------------------
    # 4. Address prefix blocks
    # ------------------------------------------------------

    for prefix_len in (
        8,
        12,
        16
    ):

        if len(addr) >= prefix_len:

            queries.append(
                (
                    """
                    SELECT *
                    FROM records
                    WHERE substr(
                        addr_n,
                        1,
                        ?
                    ) = ?
                    AND country_n = ?
                    LIMIT ?
                    """,
                    (
                        prefix_len,
                        addr[:prefix_len],
                        country,
                        100
                    )
                )
            )

    # ------------------------------------------------------
    # 5. First business-name token
    # ------------------------------------------------------

    first_name = (
        name.split()[0]
        if name
        else ""
    )

    if first_name:

        queries.append(
            (
                """
                SELECT *
                FROM records
                WHERE name_n LIKE ?
                AND country_n = ?
                LIMIT ?
                """,
                (
                    first_name + "%",
                    country,
                    100
                )
            )
        )

    # ------------------------------------------------------
    # Execute blocks independently
    # ------------------------------------------------------

    for sql, params in queries:

        for record in conn.execute(
            sql,
            params
        ):

            entity_id = record[0]

            if entity_id in seen:
                continue

            seen.add(
                entity_id
            )

            yield record

            if len(seen) >= limit:
                return


# ==========================================================
# RECORD CONVERSION
# ==========================================================

def tuple_to_record(record):

    return {
        "entity_id": record[0],
        "business_name": record[1],
        "business_address": record[2],
        "country": record[3],
        "name_n": record[4],
        "addr_n": record[5],
        "country_n": record[6]
    }


# ==========================================================
# FEATURE FUNCTIONS
# ==========================================================

def token_set(value):

    if not value:
        return set()

    return set(
        value.split()
    )


def jaccard(a, b):

    A = token_set(a)
    B = token_set(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(
        A & B
    ) / len(
        A | B
    )


def containment(a, b):

    A = token_set(a)
    B = token_set(b)

    if not A or not B:
        return 0.0

    return max(
        len(A & B) / len(A),
        len(A & B) / len(B)
    )


def digit_set(value):

    return set(
        re.findall(
            r"\d+",
            value or ""
        )
    )


def digit_overlap(a, b):

    A = digit_set(a)
    B = digit_set(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(
        A & B
    ) / len(
        A | B
    )


def first_token(value):

    return (
        value.split()[0]
        if value
        else ""
    )


def feature_row(a, b):

    an = a["name_n"]
    bn = b["name_n"]

    aa = a["addr_n"]
    ba = b["addr_n"]

    ac = a["country_n"]
    bc = b["country_n"]

    return [

        float(
            ac == bc
            and bool(ac)
        ),

        float(
            an == bn
            and bool(an)
        ),

        float(
            aa == ba
            and bool(aa)
        ),

        fuzz.ratio(
            an,
            bn
        ) / 100,

        fuzz.WRatio(
            an,
            bn
        ) / 100,

        fuzz.token_sort_ratio(
            an,
            bn
        ) / 100,

        fuzz.token_set_ratio(
            an,
            bn
        ) / 100,

        jaccard(
            an,
            bn
        ),

        containment(
            an,
            bn
        ),

        fuzz.ratio(
            aa,
            ba
        ) / 100,

        fuzz.WRatio(
            aa,
            ba
        ) / 100,

        fuzz.token_sort_ratio(
            aa,
            ba
        ) / 100,

        fuzz.token_set_ratio(
            aa,
            ba
        ) / 100,

        jaccard(
            aa,
            ba
        ),

        containment(
            aa,
            ba
        ),

        digit_overlap(
            aa,
            ba
        ),

        float(
            first_token(an)
            == first_token(bn)
            and bool(
                first_token(an)
            )
        ),

        float(
            an[:4]
            == bn[:4]
            and bool(an[:4])
        ),

        float(
            len(an)
            == len(bn)
        ),

        float(
            len(aa)
            == len(ba)
        )
    ]


FEATURE_NAMES = [

    "country_exact",

    "name_exact",

    "address_exact",

    "name_ratio",

    "name_wratio",

    "name_token_sort",

    "name_token_set",

    "name_jaccard",

    "name_containment",

    "address_ratio",

    "address_wratio",

    "address_token_sort",

    "address_token_set",

    "address_jaccard",

    "address_containment",

    "digit_overlap",

    "first_name_token_exact",

    "name_prefix4_exact",

    "name_length_equal",

    "address_length_equal"
]


# ==========================================================
# GROUND TRUTH
# ==========================================================

def load_ground_truth(path):

    result = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000
    ):

        for row in chunk.itertuples(
            index=False
        ):

            ids = set()

            if row.matched_entity_ids:

                ids = {
                    x.strip()
                    for x in (
                        row.matched_entity_ids
                        .split(",")
                    )
                    if x.strip()
                }

            result[
                row.source1_entity_id
            ] = ids

    return result


def f05(
    predicted,
    truth
):

    tp = len(
        predicted & truth
    )

    fp = len(
        predicted - truth
    )

    fn = len(
        truth - predicted
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else (
            1.0
            if not predicted
            and not truth
            else 0.0
        )
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 1.0
    )

    if (
        precision == 0
        and recall == 0
    ):
        return 0.0

    return (
        1.25
        * precision
        * recall
        /
        (
            0.25
            * precision
            + recall
        )
    )


def macro_f05(
    predictions,
    truth
):

    scores = []

    for entity_id, actual in (
        truth.items()
    ):

        predicted = predictions.get(
            entity_id,
            set()
        )

        scores.append(
            f05(
                predicted,
                actual
            )
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )