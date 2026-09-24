"""Fake GraphClient + fake LLM for testing the pipeline without real
TigerGraph or a live Anthropic API key."""
from __future__ import annotations

import json
from typing import Any, Dict, List


class FakeGraphClient:
    def __init__(self, txns: Dict[str, dict], card_windows: Dict[str, list], profiles: Dict[str, dict],
                 device_neighbors: Dict[str, dict] = None, region_histories: Dict[str, dict] = None,
                 similar_cases: List[dict] = None):
        self.txns = txns
        self.card_windows = card_windows
        self.profiles = profiles
        self.device_neighbors = device_neighbors or {}
        self.region_histories = region_histories or {}
        self._similar_cases = similar_cases or []
        self.upserted = []

    def get_transaction_details(self, txn_id):
        return self.txns[txn_id]

    def get_card_window(self, card_id, center_txn_id=None, window_minutes=0):
        return {"card_id": card_id, "transactions": self.card_windows.get(card_id, [])}

    def get_customer_profile(self, customer_id):
        return self.profiles.get(customer_id, {"customer_id": customer_id, "found": False, "channels_used": []})

    def get_device_neighbors(self, device_key):
        return self.device_neighbors.get(device_key, {"device_key": device_key, "cards": [], "customers": []})

    def get_region_cluster(self, region_code, from_ts, to_ts):
        return {"region_code": region_code, "cards": []}

    def get_card_region_history(self, card_id):
        return self.region_histories.get(card_id, {"card_id": card_id, "regions": {}})

    def get_email_neighbors(self, domain):
        return {"domain": domain, "cards": []}

    def get_velocity(self, card_id, center_txn_id, hours):
        return {"card_id": card_id, "transactions": [], "total_amt": 0.0}

    def find_similar_cases(self, embedding, top_k=5, case_source_filter=""):
        return self._similar_cases[:top_k]

    def upsert_case(self, case_record):
        self.upserted.append(case_record)
        return case_record["case_id"]


class FakeUsage:
    def __init__(self):
        self.input_tokens = 5
        self.output_tokens = 5

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class FakeLLM:
    """Returns canned JSON depending on which system prompt is used, so the
    pipeline can be exercised without network access."""

    def __init__(self, synthesis_response: dict, sar_response: dict = None):
        self.synthesis_response = synthesis_response
        self.sar_response = sar_response or {"narrative": "Test narrative.", "subjects": []}
        self.usage = FakeUsage()

    def complete(self, system, messages, max_tokens=1000, temperature=0.2):
        if "Suspicious Activity Report" in system:
            return json.dumps(self.sar_response)
        return json.dumps(self.synthesis_response)
