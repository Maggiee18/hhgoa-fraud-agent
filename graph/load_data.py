"""
Loads the four HHGOA_IEEE CSVs into a live TigerGraph instance via
pyTigerGraph's dataframe upsert API. Uses the same card_id/device_key
derivation as graph/local_backend.py (graph/derive.py) so the two backends
never disagree on an entity's identity.

Chunked to keep memory bounded against the ~591K-row transactions.csv.
Called from scripts/setup_graph.py --load.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from agent.memory import case_narrative_text, embed_text
from graph.derive import C_COLS, D_COLS, M_COLS, V_COLS, add_derived_columns, clean

DATA_DIR = os.getenv("DATA_DIR", "./data")
CHUNK = 20_000


def _row_to_vertex_attrs(row) -> dict:
    return {
        "ts": row["ts"].strftime("%Y-%m-%d %H:%M:%S"),  # explicit format, not str()'s default —
                                                          # see the MADE-edge ts fix in load_all()
        "txn_dt": int(row["TransactionDT"]) if not pd.isna(row["TransactionDT"]) else 0,
        "amt": float(row["TransactionAmt"]),
        "product_cd": clean(row.get("ProductCD")) or "",
        # clean() (not a bare `x or ""`) matters here: a missing value can arrive as
        # float('nan'), which is truthy in Python, so `nan or ""` would silently keep
        # the NaN instead of replacing it — same REST-30200 "Null JSON value" bug as
        # the card4/card6 fix above.
        "channel": clean(row.get("channel")) or "",
        "risk_score": float(row.get("risk_score")) if not pd.isna(row.get("risk_score")) else 0.0,
        "addr1": str(clean(row.get("addr1")) or ""),
        "addr2": str(clean(row.get("addr2")) or ""),
        "dist1": float(row.get("dist1")) if not pd.isna(row.get("dist1")) else 0.0,
        "dist2": float(row.get("dist2")) if not pd.isna(row.get("dist2")) else 0.0,
        "p_email_domain": clean(row.get("P_emaildomain")) or "",
        "r_email_domain": clean(row.get("R_emaildomain")) or "",
        "c_features": json.dumps({c: clean(row.get(c)) for c in C_COLS}),
        "d_features": json.dumps({c: clean(row.get(c)) for c in D_COLS}),
        "m_features": json.dumps({c: clean(row.get(c)) for c in M_COLS}),
        "v_features": json.dumps({c: clean(row.get(c)) for c in V_COLS if not pd.isna(row.get(c))}),
    }


def load_all(conn):
    print("Loading transactions.csv + identity.csv ...")
    txns = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), low_memory=False)
    identity = pd.read_csv(os.path.join(DATA_DIR, "identity.csv"), low_memory=False)
    txns = txns.merge(identity, on="TransactionID", how="left")
    txns["ts"] = pd.to_datetime(txns["ts"])

    closed_for_derive = pd.read_csv(os.path.join(DATA_DIR, "closed_cases_history.csv"), low_memory=False)
    cp_path = os.path.join(DATA_DIR, "case_pack.csv")
    case_pack_for_derive = pd.read_csv(cp_path, low_memory=False) if os.path.exists(cp_path) else None
    # card_id derivation pins ground-truth IDs from case_pack/closed_cases —
    # see graph/derive.py's module docstring for why this two-phase
    # approach exists (chronological order alone gets ~1/15 multi-card
    # customers wrong, confirmed against the real closed-case card_ids).
    txns = add_derived_columns(txns, case_pack=case_pack_for_derive, closed_cases=closed_for_derive)
    print(f"{len(txns)} transactions, {txns['card_id'].nunique()} derived cards, "
          f"{txns['customer_id'].nunique()} customers")

    # --- Customers, Cards ---
    customers = txns[["customer_id"]].drop_duplicates()
    customers["num_cards"] = customers["customer_id"].map(txns.groupby("customer_id")["card_id"].nunique())
    # v_id= already sets the primary id value; schema.gsql's
    # PRIMARY_ID_AS_ATTRIBUTE="true" is what makes it dot-referenceable in
    # queries, so it's not passed again in attributes= here (an earlier
    # attempt did, via a duplicate-named schema attribute that turned out
    # not to really exist — see docs/DECISIONS.md).
    conn.upsertVertexDataFrame(
        customers, "Customer", v_id="customer_id",
        attributes={"num_cards": "num_cards"},
    )

    cards = txns.drop_duplicates(subset=["card_id"])[["card_id", "customer_id", "card4", "card6", "card1", "card2", "card3", "card5"]].copy()
    cards = cards.rename(columns={"card4": "network", "card6": "card_type"})
    # card4/card6 (network/card_type) have missing values in the raw IEEE data like
    # every other card* column, but weren't going through the same NaN-safe clean()
    # normalization as card1/2/3/5 — a raw NaN float upserts as an invalid JSON
    # value ("Null JSON value is not supported", REST-30200). Same fix as below.
    for c in ["network", "card_type", "card1", "card2", "card3", "card5"]:
        cards[c] = cards[c].apply(lambda v: str(clean(v)) if clean(v) is not None else "")
    conn.upsertVertexDataFrame(
        cards, "Card", v_id="card_id",
        attributes={"customer_id": "customer_id", "network": "network", "card_type": "card_type",
                    "card1": "card1", "card2": "card2", "card3": "card3", "card5": "card5"},
    )
    # attributes= must be explicit here: OWNS declares no attributes in
    # schema.gsql, but `cards` carries many other columns (network,
    # card_type, card1..5, even card_id itself) and pyTigerGraph's default
    # (map every non-id column onto the edge) tried to write those onto
    # OWNS and failed ("Processing attribute card_id failed, Invalid edge
    # attribute", REST-30200). Same fix applied to every no-attribute edge
    # below.
    conn.upsertEdgeDataFrame(cards, "Customer", "OWNS", "Card", from_id="customer_id", to_id="card_id", attributes={})

    # --- Email domains, Billing regions ---
    for col, entity in [("P_emaildomain", "EmailDomain"), ("R_emaildomain", "EmailDomain")]:
        domains = txns[[col]].dropna().drop_duplicates().rename(columns={col: "domain"})
        if len(domains):
            conn.upsertVertexDataFrame(domains, "EmailDomain", v_id="domain")

    regions = txns[["addr1", "addr2"]].dropna(subset=["addr1"]).drop_duplicates(subset=["addr1"]).copy()
    regions["addr1"] = regions["addr1"].astype(str)
    regions["addr2"] = regions["addr2"].fillna("").astype(str)
    if len(regions):
        conn.upsertVertexDataFrame(regions, "BillingRegion", v_id="addr1", attributes={"country_code": "addr2"})

    # --- DeviceProfile ---
    devices = txns.dropna(subset=["device_key"]).drop_duplicates(subset=["device_key"]).copy()
    if len(devices):
        # Same REST-30200 "Null JSON value" risk as card4/card6 above — these are raw
        # identity.csv columns that can be NaN even when device_key itself is present.
        for c in ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_34", "id_23", "id_15"]:
            devices[c] = devices[c].apply(lambda v: str(clean(v)) if clean(v) is not None else "")
        devices["id_features"] = devices.apply(
            lambda r: json.dumps({c: clean(r.get(c)) for c in
                                   [f"id_{i:02d}" for i in range(1, 12)] + [f"id_{i:02d}" for i in range(12, 39)
                                                                             if i not in (15, 23, 30, 31, 33, 34)]}),
            axis=1,
        )
        conn.upsertVertexDataFrame(
            devices, "DeviceProfile", v_id="device_key",
            attributes={
                "device_type": "DeviceType", "device_info": "DeviceInfo",
                "os": "id_30", "browser": "id_31", "screen": "id_33", "match_status": "id_34",
                "proxy_flag": "id_23", "device_new": "id_15", "id_features": "id_features",
            },
        )

    # --- Transactions + edges, chunked ---
    print("Loading transactions in chunks...")
    for start in range(0, len(txns), CHUNK):
        chunk = txns.iloc[start : start + CHUNK].copy()
        attrs_df = chunk.apply(_row_to_vertex_attrs, axis=1, result_type="expand")
        attrs_df["txn_id"] = chunk["txn_id"].values
        conn.upsertVertexDataFrame(
            attrs_df, "Transaction", v_id="txn_id",
            attributes={c: c for c in attrs_df.columns if c != "txn_id"},
        )
        # MADE's ts attribute comes straight from chunk's native pandas
        # Timestamp column here, unlike Transaction.ts above (which goes
        # through _row_to_vertex_attrs's str(row["ts"])) — pyTigerGraph
        # serializes a raw Timestamp differently (likely ISO "T"-separated)
        # than the plain "YYYY-MM-DD HH:MM:SS" string TigerGraph's DATETIME
        # parser accepts, causing "Processing attribute ts failed, value
        # cannot be converted to Datetime" (REST-30200). Pre-format it the
        # same way to match.
        chunk["ts_str"] = chunk["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        conn.upsertEdgeDataFrame(chunk, "Card", "MADE", "Transaction", from_id="card_id", to_id="txn_id", attributes={"ts": "ts_str"})

        with_device = chunk.dropna(subset=["device_key"])
        if len(with_device):
            conn.upsertEdgeDataFrame(with_device, "Transaction", "FROM_DEVICE", "DeviceProfile", from_id="txn_id", to_id="device_key", attributes={})

        with_email = chunk.dropna(subset=["P_emaildomain"])
        if len(with_email):
            conn.upsertEdgeDataFrame(with_email, "Transaction", "PURCHASER_EMAIL", "EmailDomain", from_id="txn_id", to_id="P_emaildomain", attributes={})

        with_region = chunk.dropna(subset=["addr1"])
        if len(with_region):
            with_region = with_region.copy()
            with_region["addr1"] = with_region["addr1"].astype(str)
            conn.upsertEdgeDataFrame(with_region, "Transaction", "BILLED_IN", "BillingRegion", from_id="txn_id", to_id="addr1", attributes={})

        print(f"  ...{min(start + CHUNK, len(txns))}/{len(txns)}")

    # --- NEXT edges (chronological chain per card) ---
    print("Building NEXT chain per card...")
    next_edges = []
    for card_id, group in txns.groupby("card_id"):
        g = group.sort_values("ts")
        ids = g["txn_id"].tolist()
        tss = g["ts"].tolist()
        for i in range(len(ids) - 1):
            gap = int((tss[i + 1] - tss[i]).total_seconds())
            next_edges.append({"from": ids[i], "to": ids[i + 1], "gap_seconds": gap})
    if next_edges:
        next_df = pd.DataFrame(next_edges)
        conn.upsertEdgeDataFrame(next_df, "Transaction", "NEXT", "Transaction", from_id="from", to_id="to", attributes={"gap_seconds": "gap_seconds"})

    # --- ClosedCase history -> Case vertices (case_source=historical) ---
    print("Loading closed_cases_history.csv as historical Case vertices...")
    closed = closed_for_derive
    closed_attrs = []
    for _, r in closed.iterrows():
        emb = embed_text(
            case_narrative_text(
                pattern=r["pattern"], outcome_or_verdict=r["outcome"], exposure_usd=float(r["exposure_usd"]),
                n_txns=int(r["n_txns"]), analyst_notes=str(r.get("analyst_notes", "")),
            )
        )
        closed_attrs.append(
            {
                "case_id": r["case_id"], "case_source": "historical",
                "status": r["outcome"], "verdict": "fraud" if r["outcome"] == "confirmed_fraud" else "legitimate",
                "fraud_probability": 1.0 if r["outcome"] == "confirmed_fraud" else 0.0,
                "pattern": r["pattern"], "pattern_description": "",
                "exposure_usd": float(r["exposure_usd"]), "opened_at": r["opened_at"], "closed_at": r["closed_at"],
                "summary": str(r.get("analyst_notes", ""))[:500], "analyst_notes": str(r.get("analyst_notes", "")),
                "report_filed": str(r.get("report_filed", "No")).strip().lower() == "yes",
                "first_txn_id": "T" + str(r["first_fraud_txn_id"]) if not pd.isna(r["first_fraud_txn_id"]) else "",
                "embedding": emb, "card_id": r["card_id"], "customer_id": r["customer_id"],
            }
        )
    closed_df = pd.DataFrame(closed_attrs)
    conn.upsertVertexDataFrame(
        closed_df, "FraudCase", v_id="case_id",
        attributes={c: c for c in closed_df.columns if c not in ("case_id", "card_id", "customer_id")},
    )
    conn.upsertEdgeDataFrame(closed_df, "FraudCase", "ON_CARD", "Card", from_id="case_id", to_id="card_id", attributes={})

    print("Load complete.")
