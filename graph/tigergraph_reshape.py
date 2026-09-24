"""
Reshapes TigerGraph's raw installed-query result — a list of
`{PRINT-variable-name: value}` dicts, the standard pyTigerGraph/REST++
response shape — into the exact plain dicts `graph/local_backend.py`
returns, so `agent/investigation.py` / `agent/detectors.py` never need to
know which backend answered them (see `graph/client.py`'s `GraphClient`
Protocol).

Shared by `graph/tigergraph_backend.py` (direct pyTigerGraph) and
`graph/tigergraph_mcp_backend.py` (via TigerGraph MCP's
`run_installed_query` tool) — both receive byte-for-byte the same raw
shape: the MCP tool is a thin wrapper that calls
`conn.runInstalledQuery(query_name, params)` and returns its result
verbatim under `data.result` (confirmed by reading the installed
`tigergraph-mcp` package's `tools/query_tools.py` source directly — not
guessed), so one reshaping implementation serves both backends.

UNTESTED against a live TigerGraph instance (no credentials available in
this environment — see docs/DECISIONS.md's Open Blockers). Written
directly against `graph/schema/schema.gsql`'s attribute names,
`graph/gsql/queries.gsql`'s PRINT statements, and pyTigerGraph's
documented REST++ response shape: PRINT of a vertex set becomes
`[{"v_id": ..., "v_type": ..., "attributes": {...}}, ...]`; PRINT of a
scalar/accumulator becomes that value directly; `PRINT Set[Set.a, Set.b]`
(used by get_card_region_history) becomes vertex objects whose
`attributes` dict holds the projected fields, including accumulators
under their `@name`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _find(raw: List[Dict[str, Any]], name: str, default: Any = None) -> Any:
    for item in raw:
        if isinstance(item, dict) and name in item:
            return item[name]
    return default


def _vertices(raw: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    return _find(raw, name, default=[]) or []


def _attrs(vertex: Dict[str, Any]) -> Dict[str, Any]:
    return vertex.get("attributes", {}) or {}


def _v_id(vertex: Dict[str, Any]) -> Optional[str]:
    return vertex.get("v_id")


def _json_or_empty(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return {}


def reshape_transaction_details(raw: List[Dict[str, Any]], txn_id: str) -> Dict[str, Any]:
    txns = _vertices(raw, "Txn")
    if not txns:
        return {"found": False, "txn_id": txn_id}
    t = _attrs(txns[0])
    cards = _vertices(raw, "Card_")
    custs = _vertices(raw, "Cust")
    devs = _vertices(raw, "Dev")
    pemails = _vertices(raw, "PEmail")
    dev = _attrs(devs[0]) if devs else {}

    return {
        "found": True,
        "txn_id": txn_id,
        "card_id": _v_id(cards[0]) if cards else None,
        "customer_id": _v_id(custs[0]) if custs else None,
        "ts": t.get("ts"),
        "amt": t.get("amt"),
        "product_cd": t.get("product_cd") or None,
        "channel": t.get("channel"),
        "risk_score": t.get("risk_score"),
        "addr1": t.get("addr1") or None,
        "addr2": t.get("addr2") or None,
        "dist1": t.get("dist1"),
        "dist2": t.get("dist2"),
        "p_email_domain": (_v_id(pemails[0]) if pemails else None) or t.get("p_email_domain") or None,
        "r_email_domain": t.get("r_email_domain") or None,
        "device_key": _v_id(devs[0]) if devs else None,
        "device_type": dev.get("device_type"),
        "device_info": dev.get("device_info"),
        "device_new": dev.get("device_new"),
        "proxy": dev.get("proxy_flag"),  # GSQL attribute renamed to proxy_flag (PROXY is reserved); key kept as "proxy" for callers
        "os": dev.get("os"),
        "browser": dev.get("browser"),
        "screen": dev.get("screen"),
        "match_status": dev.get("match_status"),
        "c_features": _json_or_empty(t.get("c_features")),
        "d_features": _json_or_empty(t.get("d_features")),
        "m_features": _json_or_empty(t.get("m_features")),
        "v_features_nonnull_count": len(_json_or_empty(t.get("v_features"))),
    }


def reshape_card_window(raw: List[Dict[str, Any]], card_id: str) -> Dict[str, Any]:
    # Windowed == AllTxns whenever p_window_minutes == 0 (see queries.gsql),
    # which is how agent/investigation.py always calls this — so Windowed is
    # the right "transactions" result in every case this codebase makes.
    windowed = _vertices(raw, "Windowed")
    txns = []
    for v in windowed:
        a = _attrs(v)
        txns.append(
            {
                "txn_id": _v_id(v),
                "ts": a.get("ts"),
                "amt": a.get("amt"),
                "product_cd": a.get("product_cd") or None,
                "channel": a.get("channel"),
                "risk_score": a.get("risk_score"),
                "addr1": a.get("addr1") or None,
                "device_key": None,  # not a Transaction attribute; see FROM_DEVICE edge
            }
        )
    return {"card_id": card_id, "transactions": txns, "count": len(txns)}


def reshape_customer_profile(raw: List[Dict[str, Any]], customer_id: str) -> Dict[str, Any]:
    custs = _vertices(raw, "Cust")
    if not custs:
        return {"customer_id": customer_id, "found": False}
    cards = _vertices(raw, "Cards")
    cases = _vertices(raw, "Cases")
    txn_count = _find(raw, "txn_count", 0) or 0
    channels_used = _find(raw, "channels_used", []) or []
    total_amt = _find(raw, "total_amt", 0.0) or 0.0
    return {
        "customer_id": customer_id,
        "found": True,
        "cards": sorted(_v_id(c) for c in cards if _v_id(c)),
        "txn_count": int(txn_count),
        "total_amt": float(total_amt),
        "channels_used": sorted(set(channels_used)),
        "prior_closed_cases": [_v_id(c) for c in cases if _v_id(c)],
    }


def reshape_device_neighbors(raw: List[Dict[str, Any]], device_key: str) -> Dict[str, Any]:
    devs = _vertices(raw, "Dev")
    if not devs:
        return {"device_key": device_key, "cards": [], "customers": [], "txn_count": 0}
    dev = _attrs(devs[0])
    cards = _vertices(raw, "Cards")
    customers = _vertices(raw, "Customers")
    txn_count = _find(raw, "txn_count", 0) or 0
    return {
        "device_key": device_key,
        "device_info": dev.get("device_info"),
        "cards": sorted(_v_id(c) for c in cards if _v_id(c)),
        "customers": sorted(_v_id(c) for c in customers if _v_id(c)),
        "txn_count": int(txn_count),
    }


def reshape_region_cluster(raw: List[Dict[str, Any]], region_code: str) -> Dict[str, Any]:
    regions = _vertices(raw, "Region")
    if not regions:
        return {"region_code": region_code, "cards": []}
    cards = _vertices(raw, "Cards")
    txns = _vertices(raw, "Txns")
    return {
        "region_code": region_code,
        "cards": sorted(_v_id(c) for c in cards if _v_id(c)),
        "txn_count": len(txns),
    }


def reshape_card_region_history(raw: List[Dict[str, Any]], card_id: str) -> Dict[str, Any]:
    regions_list = _vertices(raw, "Regions")
    regions: Dict[str, int] = {}
    for v in regions_list:
        a = _attrs(v)
        code = a.get("region_code", _v_id(v))
        cnt = a.get("@txn_count", a.get("txn_count", 0))
        if code is not None:
            regions[str(code)] = int(cnt or 0)
    return {"card_id": card_id, "regions": regions}


def reshape_email_neighbors(raw: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
    cards = _vertices(raw, "Cards")
    txn_count = _find(raw, "txn_count", 0) or 0
    return {"domain": domain, "cards": sorted(_v_id(c) for c in cards if _v_id(c)), "txn_count": int(txn_count)}


def reshape_velocity(raw: List[Dict[str, Any]], card_id: str) -> Dict[str, Any]:
    recent = _vertices(raw, "Recent")
    total_amt = _find(raw, "total_amt", 0.0) or 0.0
    txns = [{"txn_id": _v_id(v), "ts": _attrs(v).get("ts"), "amt": _attrs(v).get("amt")} for v in recent]
    return {"card_id": card_id, "transactions": txns, "count": len(txns), "total_amt": float(total_amt)}


def reshape_similar_cases(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """find_similar_cases PRINTs a HeapAccum<ScoredCase> (TYPEDEF TUPLE<STRING
    case_id, DOUBLE score>) as @@top — a list of {"case_id":..., "score":...}
    tuple-dicts. Enrichment (pattern/outcome/exposure_usd/analyst_notes,
    case_source) happens in the caller by joining case_id against
    closed_cases_history.csv / the agent's own case store — this function
    only unwraps the raw graph result."""
    top = _find(raw, "@@top", default=None)
    if top is None:
        top = _find(raw, "top", default=[])
    results = []
    for item in top or []:
        if isinstance(item, dict) and "case_id" in item:
            results.append({"case_id": item["case_id"], "score": float(item.get("score", 0.0))})
    return results
