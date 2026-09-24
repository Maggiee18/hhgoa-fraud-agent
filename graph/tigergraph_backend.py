"""
Real TigerGraph backend, via pyTigerGraph, calling the queries installed
from graph/gsql/queries.gsql. Connection is entirely env-driven (see
.env.example) — never hardcode a host/secret here.

Only `__init__` and `_run_installed_query` talk to pyTigerGraph directly;
every public method builds the params dict (matching queries.gsql's
parameter names exactly) and reshapes the raw result via
graph/tigergraph_reshape.py into the same plain dicts local_backend.py
returns. graph/tigergraph_mcp_backend.py subclasses this and overrides only
`__init__`/`_run_installed_query`, routing the same calls through the
TigerGraph MCP server instead of a direct connection — so the two real
backends can never drift on query names, params, or result shape.

Savanna auth note (see scripts/setup_graph.py, which this mirrors): Savanna
does not accept username/password at all — auth is secret-only, tgCloud=True
+ gsqlSecret straight into the constructor. An earlier version of this file
still passed username/password (the pre-Savanna-fix idiom that setup_graph.py
already moved off of) and that was the actual cause of a "500 Internal
Server Error" on the /gsql/v1/tokens endpoint during first real validation
— not (only) the auto-suspend cold start it looked like. See
docs/DECISIONS.md.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from graph import tigergraph_reshape as reshape

load_dotenv()


def _with_wakeup_retry(fn, attempts: int = 6, delay_s: float = 20.0):
    """Savanna suspends an idle workspace; the first request after that
    wakes it but can come back as a 500/Bad-Gateway error instead of a
    clean timeout. Retry for ~2 minutes before treating it as real. Mirrors
    scripts/setup_graph.py's helper of the same name/purpose."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)
    raise last_exc  # pragma: no cover


class TigerGraphClient:
    def __init__(self):
        import pyTigerGraph as tg

        host = os.environ["TG_HOST"]
        graph_name = os.environ.get("TG_GRAPH_NAME", "HHGOA_Fraud")
        secret = os.environ.get("TG_SECRET", "")
        if not secret:
            raise RuntimeError("TG_SECRET is not set in .env — Savanna auth needs a Database Secret.")

        self.conn = tg.TigerGraphConnection(host=host, graphname=graph_name, gsqlSecret=secret, tgCloud=True)
        self.conn.apiToken = _with_wakeup_retry(lambda: self.conn.getToken(secret)[0])

    def _run_installed_query(self, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        return _with_wakeup_retry(lambda: self.conn.runInstalledQuery(name, params))

    # -- tool methods (GraphClient Protocol) ------------------------------

    def get_transaction_details(self, txn_id: str) -> Dict[str, Any]:
        raw = self._run_installed_query("get_transaction_details", {"p_txn_id": txn_id})
        return reshape.reshape_transaction_details(raw, txn_id)

    def get_card_window(
        self, card_id: str, center_txn_id: Optional[str] = None, window_minutes: int = 0
    ) -> Dict[str, Any]:
        raw = self._run_installed_query(
            "get_card_window",
            {"p_card_id": card_id, "p_center_txn_id": center_txn_id or "", "p_window_minutes": window_minutes},
        )
        return reshape.reshape_card_window(raw, card_id)

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        raw = self._run_installed_query("get_customer_profile", {"p_customer_id": customer_id})
        return reshape.reshape_customer_profile(raw, customer_id)

    def get_device_neighbors(self, device_key: str) -> Dict[str, Any]:
        raw = self._run_installed_query("get_device_neighbors", {"p_device_key": device_key})
        return reshape.reshape_device_neighbors(raw, device_key)

    def get_region_cluster(self, region_code: str, from_ts: str, to_ts: str) -> Dict[str, Any]:
        raw = self._run_installed_query(
            "get_region_cluster", {"p_region_code": region_code, "p_from": from_ts, "p_to": to_ts}
        )
        return reshape.reshape_region_cluster(raw, region_code)

    def get_card_region_history(self, card_id: str) -> Dict[str, Any]:
        raw = self._run_installed_query("get_card_region_history", {"p_card_id": card_id})
        return reshape.reshape_card_region_history(raw, card_id)

    def get_email_neighbors(self, domain: str) -> Dict[str, Any]:
        raw = self._run_installed_query("get_email_neighbors", {"p_domain": domain})
        return reshape.reshape_email_neighbors(raw, domain)

    def get_velocity(self, card_id: str, center_txn_id: str, hours: int) -> Dict[str, Any]:
        raw = self._run_installed_query(
            "get_velocity", {"p_card_id": card_id, "p_center_txn_id": center_txn_id, "p_hours": hours}
        )
        return reshape.reshape_velocity(raw, card_id)

    def find_similar_cases(
        self, embedding: List[float], top_k: int = 5, case_source_filter: str = ""
    ) -> List[Dict[str, Any]]:
        raw = self._run_installed_query(
            "find_similar_cases",
            {"p_embedding": embedding, "p_top_k": top_k, "p_case_source_filter": case_source_filter},
        )
        results = reshape.reshape_similar_cases(raw)
        return _enrich_similar_cases(results)

    def upsert_case(self, case_record: Dict[str, Any]) -> str:
        self._run_installed_query(
            "upsert_case",
            {
                "p_case_id": case_record["case_id"],
                "p_case_source": case_record.get("case_source", "agent"),
                "p_status": case_record["status"],
                "p_verdict": case_record["verdict"],
                "p_fraud_probability": case_record["fraud_probability"],
                "p_pattern": case_record["pattern"],
                "p_pattern_description": case_record.get("pattern_description", ""),
                "p_exposure_usd": case_record.get("exposure_usd", 0.0),
                "p_opened_at": case_record["opened_at"],
                "p_closed_at": case_record.get("closed_at", case_record["opened_at"]),
                "p_summary": case_record.get("summary", ""),
                "p_analyst_notes": case_record.get("analyst_notes", ""),
                "p_report_filed": case_record.get("report_filed", False),
                "p_first_txn_id": case_record.get("first_txn_id", ""),
                "p_embedding": case_record.get("embedding", []),
                "p_txn_ids": case_record.get("txn_ids", []),
                "p_card_ids": case_record.get("card_ids", []),
                "p_connected_card_ids": case_record.get("connected_card_ids", []),
                "p_device_keys": case_record.get("device_keys", []),
            },
        )
        return case_record["case_id"]


_closed_cases_cache = None


def _enrich_similar_cases(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """find_similar_cases's GSQL only returns (case_id, score) — join against
    closed_cases_history.csv for the pattern/outcome/exposure/analyst_notes
    context agent/investigation.py's LLM synthesis payload uses (mirrors
    what local_backend.py returns inline). Any case_id not found there is an
    agent-written case already in the graph but not the CSV — kept with
    just case_id/score/case_source="agent" rather than dropped."""
    global _closed_cases_cache
    if not results:
        return results
    if _closed_cases_cache is None:
        import pandas as pd

        path = os.path.join(os.getenv("DATA_DIR", "./data"), "closed_cases_history.csv")
        _closed_cases_cache = pd.read_csv(path, low_memory=False) if os.path.exists(path) else None

    enriched = []
    for r in results:
        row = None
        if _closed_cases_cache is not None:
            match = _closed_cases_cache[_closed_cases_cache["case_id"] == r["case_id"]]
            if len(match):
                row = match.iloc[0]
        if row is not None:
            enriched.append(
                {
                    "case_id": r["case_id"],
                    "case_source": "historical",
                    "score": r["score"],
                    "pattern": row["pattern"],
                    "outcome": row["outcome"],
                    "exposure_usd": float(row["exposure_usd"]),
                    "analyst_notes": row.get("analyst_notes", ""),
                }
            )
        else:
            enriched.append({"case_id": r["case_id"], "case_source": "agent", "score": r["score"]})
    return enriched
