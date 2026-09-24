"""
Shared derivation logic used by both graph/local_backend.py (dev/testing)
and graph/load_data.py (real TigerGraph load), so the two backends can
never drift into assigning different card_ids/device_keys for the same
data.

card_id derivation — validated against the full dataset (see
docs/DECISIONS.md for the investigation): a customer's several cards are
NOT reliably separable by chronological first-use order alone. Checking
against every closed-case and case-pack card_id confirmed the tuple
(card1..card6) does identify a distinct card, but the K-index (K1 vs K2 vs
...) that the challenge authors assigned does not always follow first-use
order — in one confirmed example a card used in 1,134 transactions is
labeled K2 while a 6-transaction variant of the same card1 (with the other
fields blank) is K1. There is no way to recover their exact original
numbering scheme from the columns provided.

The fix that IS fully correct: case_pack.csv and closed_cases_history.csv
already tell us the ground-truth card_id for every card they reference,
via a known transaction (flagged_txn_id / first_fraud_txn_id). We look up
that transaction's (card1..card6) tuple and pin it to the given card_id.
Every other tuple for that customer gets numbered into the remaining K-slots.
Customers that never appear in either reference file (the large majority —
these never need to be scored against a known card_id) fall back to fast
chronological-first-use numbering, which is a reasonable, clearly-labeled
best effort and never collides with a pinned number.

Validated: 0/1913 closed-case card_ids missing, 0/20 case_pack flagged-
transaction card_id mismatches, after this two-phase derivation.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]
DEVICE_COLS = ["DeviceInfo", "DeviceType", "id_30", "id_31", "id_33", "id_34"]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
V_COLS = [f"V{i}" for i in range(1, 340)]
ID_NUMERIC_COLS = [f"id_{i:02d}" for i in range(1, 12)]
ID_CATEGORICAL_COLS = [f"id_{i:02d}" for i in range(12, 39)]


def clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def device_key(row) -> Optional[str]:
    if pd.isna(row.get("DeviceInfo")) and pd.isna(row.get("id_30")) and pd.isna(row.get("DeviceType")):
        return None
    key_str = "|".join(str(row.get(c, "")) for c in DEVICE_COLS)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _card_tuple(row) -> Tuple:
    return tuple(clean(row.get(c)) for c in CARD_COLS)


def build_known_card_registry(
    case_pack: Optional[pd.DataFrame], closed_cases: Optional[pd.DataFrame], txns: pd.DataFrame
) -> Dict[Tuple[str, Tuple], str]:
    """(customer_id, card_tuple) -> ground-truth card_id, for every card_id
    referenced in case_pack.csv (via flagged_txn_id) or
    closed_cases_history.csv (via first_fraud_txn_id)."""
    registry: Dict[Tuple[str, Tuple], str] = {}
    if "tup" not in txns.columns:
        raise ValueError("txns must have a 'tup' column (see add_derived_columns)")
    by_txn_id = txns.drop_duplicates(subset=["TransactionID"]).set_index("TransactionID")

    if case_pack is not None:
        for _, r in case_pack.iterrows():
            tid = r.get("flagged_txn_id")
            if pd.notna(tid) and int(tid) in by_txn_id.index:
                row = by_txn_id.loc[int(tid)]
                registry[(r["customer_id"], row["tup"])] = r["card_id"]

    if closed_cases is not None:
        for _, r in closed_cases.iterrows():
            tid = r.get("first_fraud_txn_id")
            if pd.notna(tid) and int(tid) in by_txn_id.index:
                row = by_txn_id.loc[int(tid)]
                registry[(r["customer_id"], row["tup"])] = r["card_id"]

    return registry


def add_derived_columns(
    txns: pd.DataFrame,
    case_pack: Optional[pd.DataFrame] = None,
    closed_cases: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Adds device_key, card_id, txn_id to a transactions dataframe (already
    merged with identity.csv on TransactionID, with ts parsed). Mutates and
    returns the same dataframe, sorted by (customer_id, ts).

    Pass case_pack/closed_cases (the dataframes read from those CSVs) so the
    known-card registry can pin the ground-truth card_id for every
    referenced card — strongly recommended; without them every customer
    falls back to the chronological-order heuristic, which is wrong for
    ~1 in 15 multi-card customers (see module docstring)."""
    txns = txns.sort_values(["customer_id", "ts"]).reset_index(drop=True)
    txns["device_key"] = txns.apply(device_key, axis=1)
    txns["tup"] = txns.apply(_card_tuple, axis=1)

    # fast default: chronological-first-use numbering for everyone
    txns["card_rank"] = txns.groupby("customer_id")["tup"].transform(lambda s: pd.factorize(s)[0] + 1)
    txns["card_id"] = txns["customer_id"] + "-K" + txns["card_rank"].astype(str)

    registry = build_known_card_registry(case_pack, closed_cases, txns)
    if registry:
        affected_customers = {k[0] for k in registry.keys()}
        mask = txns["customer_id"].isin(affected_customers)
        final_card_id = txns["card_id"].copy()
        for customer_id, idxs in txns[mask].groupby("customer_id").groups.items():
            idxs = list(idxs)
            tuples_in_order, seen = [], set()
            for i in idxs:
                tup = txns.at[i, "tup"]
                if tup not in seen:
                    seen.add(tup)
                    tuples_in_order.append(tup)
            pinned: Dict[Tuple, str] = {}
            used_k = set()
            for tup in tuples_in_order:
                cid = registry.get((customer_id, tup))
                if cid:
                    pinned[tup] = cid
                    used_k.add(int(cid.split("-K")[1]))
            next_k = 1
            tup_to_card = {}
            for tup in tuples_in_order:
                if tup in pinned:
                    tup_to_card[tup] = pinned[tup]
                else:
                    while next_k in used_k:
                        next_k += 1
                    tup_to_card[tup] = f"{customer_id}-K{next_k}"
                    used_k.add(next_k)
            for i in idxs:
                final_card_id.at[i] = tup_to_card[txns.at[i, "tup"]]
        txns["card_id"] = final_card_id

    txns = txns.drop(columns=["tup", "card_rank"])
    txns["txn_id"] = "T" + txns["TransactionID"].astype(str)
    return txns


def validate_card_ids(txns: pd.DataFrame, closed_cases: pd.DataFrame) -> Dict[str, int]:
    derived_cards = set(txns["card_id"].dropna().unique())
    missing = sum(1 for cid in closed_cases["card_id"].dropna().unique() if cid not in derived_cards)
    return {
        "closed_case_cards_checked": int(closed_cases["card_id"].nunique()),
        "missing_from_derivation": missing,
    }
