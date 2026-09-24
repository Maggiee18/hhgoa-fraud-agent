"""
Local (pandas-indexed) implementation of the GraphClient interface — see
graph/client.py for why this exists alongside the TigerGraph backend.

card_id derivation: the README states customer_id is "derived from the card
issuer field" and that "one customer can have several cards" labeled
`C01234-K1`, `C01234-K2`, ... but does not give the exact rule for splitting
a customer's transactions into distinct cards. This module derives it the
same way any reasonable engineer would from the raw signal actually present
(card1..card6 = issuer/network/type columns): group a customer's
transactions by the distinct (card1..card6) tuple, and number the tuples
K1, K2, ... in order of first use. This is documented as an assumption in
docs/DECISIONS.md and cross-checked at load time against
closed_cases_history.csv's real card_ids (which must already exist in this
derivation, or the assumption is wrong and needs revisiting).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from graph.derive import C_COLS, D_COLS, M_COLS, V_COLS, add_derived_columns, clean, validate_card_ids

DATA_DIR = os.getenv("DATA_DIR", "./data")
_clean = clean


class LocalGraphClient:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DATA_DIR
        self._load()
        self._agent_cases: Dict[str, Dict[str, Any]] = {}

    # -- loading -----------------------------------------------------

    def _load(self):
        txn_path = os.path.join(self.data_dir, "transactions.csv")
        id_path = os.path.join(self.data_dir, "identity.csv")
        cc_path = os.path.join(self.data_dir, "closed_cases_history.csv")
        cp_path = os.path.join(self.data_dir, "case_pack.csv")

        txns = pd.read_csv(txn_path, low_memory=False)
        identity = pd.read_csv(id_path, low_memory=False)
        txns = txns.merge(identity, on="TransactionID", how="left")
        txns["ts"] = pd.to_datetime(txns["ts"])

        self.closed_cases = pd.read_csv(cc_path, low_memory=False)
        case_pack = pd.read_csv(cp_path, low_memory=False) if os.path.exists(cp_path) else None

        # card_id derivation needs case_pack/closed_cases to pin ground-truth
        # IDs — see graph/derive.py's module docstring for why.
        txns = add_derived_columns(txns, case_pack=case_pack, closed_cases=self.closed_cases)

        self.txns = txns.set_index("txn_id", drop=False)
        self._by_card = txns.set_index("card_id", drop=False).sort_index()
        self._by_customer = txns.set_index("customer_id", drop=False).sort_index()
        self._by_device = txns.dropna(subset=["device_key"]).set_index("device_key", drop=False).sort_index()
        self._by_region = txns.dropna(subset=["addr1"]).set_index("addr1", drop=False).sort_index()

        self.closed_cases["opened_at"] = pd.to_datetime(self.closed_cases["opened_at"])
        self.closed_cases["closed_at"] = pd.to_datetime(self.closed_cases["closed_at"])

        # validate the card_id derivation against real closed-case card_ids
        self._card_id_validation = validate_card_ids(txns, self.closed_cases)

        self._embed_cache: Dict[str, List[float]] = {}
        self._closed_case_embeddings: Optional[np.ndarray] = None
        self._closed_case_ids: Optional[List[str]] = None

    # -- tool methods --------------------------------------------------

    def get_transaction_details(self, txn_id: str) -> Dict[str, Any]:
        if txn_id not in self.txns.index:
            return {"found": False, "txn_id": txn_id}
        row = self.txns.loc[txn_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        c_features = {c: _clean(row.get(c)) for c in C_COLS}
        d_features = {c: _clean(row.get(c)) for c in D_COLS}
        m_features = {c: _clean(row.get(c)) for c in M_COLS}
        v_features = {c: _clean(row.get(c)) for c in V_COLS if not pd.isna(row.get(c))}
        return {
            "found": True,
            "txn_id": txn_id,
            "card_id": row["card_id"],
            "customer_id": row["customer_id"],
            "ts": str(row["ts"]),
            "amt": float(row["TransactionAmt"]),
            "product_cd": _clean(row.get("ProductCD")),
            "channel": row.get("channel"),
            "risk_score": _clean(row.get("risk_score")),
            "addr1": _clean(row.get("addr1")),
            "addr2": _clean(row.get("addr2")),
            "dist1": _clean(row.get("dist1")),
            "dist2": _clean(row.get("dist2")),
            "p_email_domain": _clean(row.get("P_emaildomain")),
            "r_email_domain": _clean(row.get("R_emaildomain")),
            "device_key": _clean(row.get("device_key")),
            "device_type": _clean(row.get("DeviceType")),
            "device_info": _clean(row.get("DeviceInfo")),
            "device_new": _clean(row.get("id_15")),
            "proxy": _clean(row.get("id_23")),
            "os": _clean(row.get("id_30")),
            "browser": _clean(row.get("id_31")),
            "screen": _clean(row.get("id_33")),
            "match_status": _clean(row.get("id_34")),
            "c_features": c_features,
            "d_features": d_features,
            "m_features": m_features,
            "v_features_nonnull_count": len(v_features),
        }

    def get_card_window(
        self, card_id: str, center_txn_id: Optional[str] = None, window_minutes: int = 0
    ) -> Dict[str, Any]:
        if card_id not in self._by_card.index:
            return {"card_id": card_id, "transactions": []}
        rows = self._by_card.loc[[card_id]] if card_id in self._by_card.index else self._by_card.iloc[0:0]
        rows = rows.sort_values("ts")
        if center_txn_id and window_minutes:
            if center_txn_id in self.txns.index:
                center_ts = self.txns.loc[center_txn_id]["ts"]
                lo = center_ts - pd.Timedelta(minutes=window_minutes)
                hi = center_ts + pd.Timedelta(minutes=window_minutes)
                rows = rows[(rows["ts"] >= lo) & (rows["ts"] <= hi)]
        txns = [
            {
                "txn_id": r["txn_id"],
                "ts": str(r["ts"]),
                "amt": float(r["TransactionAmt"]),
                "product_cd": _clean(r.get("ProductCD")),
                "channel": r.get("channel"),
                "risk_score": _clean(r.get("risk_score")),
                "addr1": _clean(r.get("addr1")),
                "device_key": _clean(r.get("device_key")),
            }
            for _, r in rows.iterrows()
        ]
        return {"card_id": card_id, "transactions": txns, "count": len(txns)}

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        if customer_id not in self._by_customer.index:
            return {"customer_id": customer_id, "found": False}
        rows = self._by_customer.loc[[customer_id]]
        cards = sorted(rows["card_id"].dropna().unique().tolist())
        cases = self.closed_cases[self.closed_cases["customer_id"] == customer_id]
        return {
            "customer_id": customer_id,
            "found": True,
            "cards": cards,
            "txn_count": int(len(rows)),
            "total_amt": float(rows["TransactionAmt"].sum()),
            "channels_used": sorted(rows["channel"].dropna().unique().tolist()),
            "prior_closed_cases": cases["case_id"].tolist(),
        }

    def get_device_neighbors(self, device_key: str) -> Dict[str, Any]:
        if not device_key or device_key not in self._by_device.index:
            return {"device_key": device_key, "cards": [], "customers": [], "txn_count": 0}
        rows = self._by_device.loc[[device_key]]
        return {
            "device_key": device_key,
            "device_info": _clean(rows.iloc[0].get("DeviceInfo")),
            "cards": sorted(rows["card_id"].dropna().unique().tolist()),
            "customers": sorted(rows["customer_id"].dropna().unique().tolist()),
            "txn_count": int(len(rows)),
        }

    def get_region_cluster(self, region_code: str, from_ts: str, to_ts: str) -> Dict[str, Any]:
        if region_code not in self._by_region.index:
            return {"region_code": region_code, "cards": []}
        rows = self._by_region.loc[[region_code]]
        if from_ts and to_ts:
            rows = rows[(rows["ts"] >= pd.Timestamp(from_ts)) & (rows["ts"] <= pd.Timestamp(to_ts))]
        return {
            "region_code": region_code,
            "cards": sorted(rows["card_id"].dropna().unique().tolist()),
            "txn_count": int(len(rows)),
        }

    def get_card_region_history(self, card_id: str) -> Dict[str, Any]:
        if card_id not in self._by_card.index:
            return {"card_id": card_id, "regions": {}}
        rows = self._by_card.loc[[card_id]]
        counts = rows["addr1"].value_counts(dropna=True).to_dict()
        return {"card_id": card_id, "regions": {str(k): int(v) for k, v in counts.items()}}

    def get_email_neighbors(self, domain: str) -> Dict[str, Any]:
        rows = self.txns[self.txns["P_emaildomain"] == domain]
        return {
            "domain": domain,
            "cards": sorted(rows["card_id"].dropna().unique().tolist()),
            "txn_count": int(len(rows)),
        }

    def get_velocity(self, card_id: str, center_txn_id: str, hours: int) -> Dict[str, Any]:
        if card_id not in self._by_card.index or center_txn_id not in self.txns.index:
            return {"card_id": card_id, "transactions": [], "total_amt": 0.0}
        center_ts = self.txns.loc[center_txn_id]["ts"]
        rows = self._by_card.loc[[card_id]]
        rows = rows[(rows["ts"] <= center_ts) & (rows["ts"] >= center_ts - pd.Timedelta(hours=hours))]
        rows = rows.sort_values("ts")
        txns = [
            {"txn_id": r["txn_id"], "ts": str(r["ts"]), "amt": float(r["TransactionAmt"])}
            for _, r in rows.iterrows()
        ]
        return {
            "card_id": card_id,
            "transactions": txns,
            "count": len(txns),
            "total_amt": float(rows["TransactionAmt"].sum()),
        }

    # -- case memory -----------------------------------------------------

    def _ensure_closed_case_embeddings(self):
        if self._closed_case_embeddings is not None:
            return
        from agent.memory import case_narrative_text, embed_text

        texts = []
        ids = []
        for _, r in self.closed_cases.iterrows():
            texts.append(
                case_narrative_text(
                    pattern=r["pattern"],
                    outcome_or_verdict=r["outcome"],
                    exposure_usd=float(r["exposure_usd"]),
                    n_txns=int(r["n_txns"]),
                    analyst_notes=str(r.get("analyst_notes", "")),
                )
            )
            ids.append(r["case_id"])
        embeddings = [embed_text(t) for t in texts]
        self._closed_case_embeddings = np.array(embeddings)
        self._closed_case_ids = ids

    def find_similar_cases(
        self, embedding: List[float], top_k: int = 5, case_source_filter: str = ""
    ) -> List[Dict[str, Any]]:
        results = []
        q = np.array(embedding)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        if case_source_filter != "agent":
            self._ensure_closed_case_embeddings()
            mat = self._closed_case_embeddings
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            sims = (mat / norms) @ q_norm
            order = np.argsort(-sims)[:top_k]
            for i in order:
                cid = self._closed_case_ids[i]
                row = self.closed_cases[self.closed_cases["case_id"] == cid].iloc[0]
                results.append(
                    {
                        "case_id": cid,
                        "case_source": "historical",
                        "score": float(sims[i]),
                        "pattern": row["pattern"],
                        "outcome": row["outcome"],
                        "exposure_usd": float(row["exposure_usd"]),
                        "analyst_notes": row["analyst_notes"],
                    }
                )

        if case_source_filter != "historical":
            for cid, rec in self._agent_cases.items():
                emb = np.array(rec.get("embedding", []))
                if emb.size == 0:
                    continue
                sim = float(emb / (np.linalg.norm(emb) + 1e-9) @ q_norm)
                results.append({"case_id": cid, "case_source": "agent", "score": sim, **rec.get("summary_fields", {})})

        results.sort(key=lambda r: -r["score"])
        return results[:top_k]

    def upsert_case(self, case_record: Dict[str, Any]) -> str:
        case_id = case_record["case_id"]
        self._agent_cases[case_id] = case_record
        # mirror to disk so it survives process restarts and is inspectable
        os.makedirs(os.path.join(self.data_dir, "..", "graph_mirror"), exist_ok=True)
        path = os.path.join(self.data_dir, "..", "graph_mirror", f"{case_id}.json")
        with open(path, "w") as f:
            json.dump({k: v for k, v in case_record.items() if k != "embedding"}, f, default=str, indent=2)
        return case_id
