"""Loading the IBM Transactions for AML dataset, plus a synthetic generator (spec §6.1).

The IBM-AML transaction CSV (e.g. ``LI-Small_Trans.csv``) has the columns:

    Timestamp, From Bank, Account, To Bank, Account.1,
    Amount Received, Receiving Currency, Amount Paid, Payment Currency,
    Payment Format, Is Laundering

An account hex id is only unique *within* a bank, so the canonical account key is the
``(bank, account)`` pair (see ``account_key``). ``Is Laundering`` is the per-transaction label
(0/1); positives are rare (~0.05-0.1% for the LI variants).

``make_synthetic_aml`` emits a small DataFrame with the *same schema* and a handful of injected
laundering typologies, so the whole pipeline can be smoke-tested on the login node (CPU, no
download) before the real data lands. Real and synthetic frames are interchangeable downstream.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

# Canonical column names as shipped by the IBM-AML Kaggle CSVs.
COL_TIMESTAMP = "Timestamp"
COL_FROM_BANK = "From Bank"
COL_FROM_ACCT = "Account"
COL_TO_BANK = "To Bank"
COL_TO_ACCT = "Account.1"
COL_AMT_RECEIVED = "Amount Received"
COL_CUR_RECEIVED = "Receiving Currency"
COL_AMT_PAID = "Amount Paid"
COL_CUR_PAID = "Payment Currency"
COL_PAYMENT_FORMAT = "Payment Format"
COL_IS_LAUNDERING = "Is Laundering"

REQUIRED_COLUMNS = [
    COL_TIMESTAMP, COL_FROM_BANK, COL_FROM_ACCT, COL_TO_BANK, COL_TO_ACCT,
    COL_AMT_RECEIVED, COL_CUR_RECEIVED, COL_AMT_PAID, COL_CUR_PAID,
    COL_PAYMENT_FORMAT, COL_IS_LAUNDERING,
]

# IBM-AML timestamps look like "2022/09/01 00:08".
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def account_key(bank: pd.Series, account: pd.Series) -> pd.Series:
    """Stable per-bank-unique account identifier, e.g. ``"012->8A1F3C"`` -> ``"12_8A1F3C"``."""
    return bank.astype(str).str.strip() + "_" + account.astype(str).str.strip()


def normalize(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Validate the schema and add derived columns used by graph construction.

    Adds:
      - ``t``            integer seconds since the earliest transaction (parsed from Timestamp)
      - ``src_account``  canonical sender account key   (From Bank, Account)
      - ``dst_account``  canonical receiver account key  (To Bank, Account.1)
      - ``label``        int 0/1 copy of ``Is Laundering``

    ``copy=False`` mutates the input in place — value-identical to ``copy=True`` but avoids a full
    duplication of the (large) frame; callers that own a freshly-loaded frame pass it to roughly
    halve peak memory on the real 6.9M-row dataset.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required IBM-AML columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if copy:
        df = df.copy()
    ts = pd.to_datetime(df[COL_TIMESTAMP], format=TIMESTAMP_FORMAT, errors="coerce")
    if ts.isna().any():
        # Fall back to a flexible parse for any rows the strict format missed.
        ts = ts.fillna(pd.to_datetime(df[COL_TIMESTAMP], errors="coerce"))
    if ts.isna().any():
        n_bad = int(ts.isna().sum())
        raise ValueError(f"Could not parse {n_bad} Timestamp value(s); check the input format.")

    t0 = ts.min()
    df["t"] = (ts - t0).dt.total_seconds().astype("int64")
    df["src_account"] = account_key(df[COL_FROM_BANK], df[COL_FROM_ACCT])
    df["dst_account"] = account_key(df[COL_TO_BANK], df[COL_TO_ACCT])
    df["label"] = df[COL_IS_LAUNDERING].astype(int)

    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    return df


def load_ibm_aml(trans_csv: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load and normalize an IBM-AML transactions CSV.

    Parameters
    ----------
    trans_csv : path to e.g. ``data/raw/LI-Small_Trans.csv``.
    nrows     : optional row cap for quick iteration on a slice.
    """
    trans_csv = Path(trans_csv)
    if not trans_csv.exists():
        raise FileNotFoundError(
            f"Transactions CSV not found: {trans_csv}\n"
            f"Download it first: bash data/download_ibm_aml.sh (needs a Kaggle token), "
            f"or run the pipeline with --synthetic for a smoke test."
        )
    # The two high-cardinality account columns dominate read memory; load them as `category`
    # (they only feed account_key -> node identity, not the feature encodings, so the graph and
    # features stay byte-identical). Currencies/format/amounts are left at default dtypes on
    # purpose — categorising those shifts feature values vs the trained checkpoint.
    acct_dtypes = {COL_FROM_ACCT: "category", COL_TO_ACCT: "category"}
    df = pd.read_csv(trans_csv, nrows=nrows, dtype=acct_dtypes)
    return normalize(df, copy=False)   # we own this freshly-read frame; skip the duplicate copy


def make_synthetic_aml(
    n_accounts: int = 200,
    n_legit: int = 4000,
    n_banks: int = 4,
    seed: int = 0,
    start: str = "2022/09/01 00:00",
    return_pattern_labels: bool = False,
):
    """Generate a tiny IBM-AML-schema DataFrame with injected laundering typologies.

    Produces mostly legitimate transactions plus a few classic structures (fan-out / smurfing,
    layering chain, circular flow) whose transactions are labelled ``Is Laundering = 1``. The
    result is schema-identical to the real CSV, so the full pipeline can be exercised offline.
    Not a realistic simulator — it exists only to smoke-test graph construction.

    With ``return_pattern_labels=True`` returns ``(df, pattern_per_row)`` where pattern_per_row is a
    string array aligned to the normalized (node) order, each in
    {"legit","fan_out","layering_chain","circular"} — the ground-truth typology oracle for tests.
    """
    rng = random.Random(seed)
    start_ts = pd.Timestamp(start)
    currencies = ["US Dollar", "Euro", "Yuan", "Bitcoin"]
    formats = ["Cheque", "Credit Card", "ACH", "Wire", "Reinvestment", "Cash"]

    accounts = [(rng.randrange(n_banks), f"{rng.randrange(16**6):06X}") for _ in range(n_accounts)]

    rows: list[dict] = []

    def add(src, dst, minute, amount, illicit, pattern="legit"):
        cur = rng.choice(currencies)
        rows.append({
            COL_TIMESTAMP: (start_ts + pd.Timedelta(minutes=minute)).strftime(TIMESTAMP_FORMAT),
            COL_FROM_BANK: src[0], COL_FROM_ACCT: src[1],
            COL_TO_BANK: dst[0], COL_TO_ACCT: dst[1],
            COL_AMT_RECEIVED: round(amount, 2), COL_CUR_RECEIVED: cur,
            COL_AMT_PAID: round(amount, 2), COL_CUR_PAID: cur,
            COL_PAYMENT_FORMAT: rng.choice(formats),
            COL_IS_LAUNDERING: int(illicit),
            "_pattern": pattern,
        })

    # Legitimate background traffic spread over ~30 days.
    horizon = 30 * 24 * 60
    for _ in range(n_legit):
        src, dst = rng.sample(accounts, 2)
        add(src, dst, rng.randrange(horizon), rng.uniform(10, 5000), illicit=False)

    # Injected laundering patterns, clustered in time so the Δt flow edges connect them.
    def fan_out(base_minute):
        source = rng.choice(accounts)
        mules = rng.sample(accounts, 8)
        for i, m in enumerate(mules):
            add(source, m, base_minute + i, rng.uniform(8000, 9000), illicit=True, pattern="fan_out")

    def layering_chain(base_minute):
        chain = rng.sample(accounts, 6)
        for i in range(len(chain) - 1):
            add(chain[i], chain[i + 1], base_minute + i * 3, rng.uniform(9000, 11000),
                illicit=True, pattern="layering_chain")

    def circular(base_minute):
        cyc = rng.sample(accounts, 5)
        for i in range(len(cyc)):
            add(cyc[i], cyc[(i + 1) % len(cyc)], base_minute + i * 2, rng.uniform(11000, 13000),
                illicit=True, pattern="circular")

    for k in range(6):
        fan_out(rng.randrange(horizon))
        layering_chain(rng.randrange(horizon))
        circular(rng.randrange(horizon))

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["_pattern"])
    df = normalize(df)                                  # sorts by time; _pattern rides along
    patterns = df["_pattern"].to_numpy()
    df = df.drop(columns=["_pattern"])
    if return_pattern_labels:
        return df, patterns
    return df
