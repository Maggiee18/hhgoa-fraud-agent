"""
Graph access layer: one interface, three backends.

- `tigergraph`: pyTigerGraph against a real Savanna/CE instance, calling the
  installed queries in graph/gsql/queries.gsql.
- `tigergraph_mcp`: the same graph and the same installed queries, routed
  through the TigerGraph MCP server instead of a direct connection — what
  satisfies the challenge's "use TigerGraph MCP" required component
  literally (graph/tigergraph_mcp_backend.py). Either this or `tigergraph`
  is the required target for submission.
- `local`: an in-memory pandas-indexed store built directly from the same
  CSVs, implementing the identical methods. Exists because this development
  environment's network cannot reach *.tgcloud.io (see docs/DECISIONS.md) —
  it lets the agent be built, tested, and demoed end-to-end without live
  infra, and serves as an offline fallback. It is NOT a shortcut on what
  gets built for TigerGraph; it is the same query surface, swappable via
  the GRAPH_BACKEND env var once real credentials are available.

Every tool in agent/ talks to `GraphClient`, never to a backend directly,
so swapping backends changes zero agent code.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol


class GraphClient(Protocol):
    def get_transaction_details(self, txn_id: str) -> Dict[str, Any]: ...

    def get_card_window(
        self, card_id: str, center_txn_id: Optional[str] = None, window_minutes: int = 0
    ) -> Dict[str, Any]: ...

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]: ...

    def get_device_neighbors(self, device_key: str) -> Dict[str, Any]: ...

    def get_region_cluster(self, region_code: str, from_ts: str, to_ts: str) -> Dict[str, Any]: ...

    def get_card_region_history(self, card_id: str) -> Dict[str, Any]: ...

    def get_email_neighbors(self, domain: str) -> Dict[str, Any]: ...

    def get_velocity(self, card_id: str, center_txn_id: str, hours: int) -> Dict[str, Any]: ...

    def find_similar_cases(
        self, embedding: List[float], top_k: int = 5, case_source_filter: str = ""
    ) -> List[Dict[str, Any]]: ...

    def upsert_case(self, case_record: Dict[str, Any]) -> str: ...


_client: Optional[GraphClient] = None


def get_graph_client(force_backend: Optional[str] = None) -> GraphClient:
    global _client
    backend = force_backend or os.getenv("GRAPH_BACKEND", "local")
    if _client is not None and force_backend is None:
        return _client

    if backend == "tigergraph":
        from graph.tigergraph_backend import TigerGraphClient

        client: GraphClient = TigerGraphClient()
    elif backend == "tigergraph_mcp":
        from graph.tigergraph_mcp_backend import TigerGraphMCPClient

        client = TigerGraphMCPClient()
    elif backend == "local":
        from graph.local_backend import LocalGraphClient

        client = LocalGraphClient()
    else:
        raise ValueError(f"Unknown GRAPH_BACKEND: {backend}")

    if force_backend is None:
        _client = client
    return client
