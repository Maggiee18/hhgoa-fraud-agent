"""
The investigation pipeline: gather evidence from the graph, run
deterministic pattern detectors, retrieve case memory, synthesize with the
LLM, decide actions via the policy engine, simulate any requested evidence
response, reassess, and produce the final CaseAnswer.

This module is intentionally LLM-call-isolated: every function except
`synthesize_with_llm` and `write_sar_narrative` is pure/deterministic and
unit-testable with a mock GraphClient and no API key (see tests/).
agent/graph_agent.py wraps this pipeline as a LangGraph state graph for the
explicit loop-with-stopping-criteria the spec asks for; the logic itself
lives here so it's easy to test and reason about independent of LangGraph.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agent import memory, policy
from agent.detectors import PatternSignal, run_all_detectors
from agent.schemas import (
    ActionRecommendation,
    Case,
    CaseAnswer,
    Evidence,
    EvidenceRequest,
    EvidenceRequestType,
    NextBestActions,
    Pattern,
    SAR,
    Verdict,
)
from agent.uncertainty import UncertaintyAssessment, assess

MAX_EVIDENCE_ROUNDS = 2  # explicit stopping guard against infinite loops


@dataclass
class EvidenceBundle:
    txn: Dict[str, Any]
    card_window: Dict[str, Any]
    card_region_history: Dict[str, Any]
    customer_profile: Dict[str, Any]
    device_neighbors: Optional[Dict[str, Any]] = None
    region_cluster: Optional[Dict[str, Any]] = None
    tool_calls: int = 0


@dataclass
class InvestigationState:
    case_id: str
    trigger_type: str
    trigger_text: str
    flagged_txn_id: str
    card_id: str
    customer_id: str
    risk_score: Optional[float]
    step: int = 0
    evidence: Optional[EvidenceBundle] = None
    signals: List[PatternSignal] = field(default_factory=list)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    llm_synthesis: Dict[str, Any] = field(default_factory=dict)
    evidence_requests: List[EvidenceRequest] = field(default_factory=list)
    customer_response: Optional[str] = None
    initial_actions: List[ActionRecommendation] = field(default_factory=list)
    final_actions: List[ActionRecommendation] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None
    tool_calls: int = 0
    tokens: int = 0
    t0: float = field(default_factory=time.time)


# --- 1. Evidence gathering -----------------------------------------------


def gather_evidence(state: InvestigationState, graph_client) -> EvidenceBundle:
    calls = 0
    txn = graph_client.get_transaction_details(state.flagged_txn_id)
    calls += 1
    card_window = graph_client.get_card_window(state.card_id, state.flagged_txn_id, window_minutes=0)
    calls += 1
    card_region_history = graph_client.get_card_region_history(state.card_id)
    calls += 1
    customer_profile = graph_client.get_customer_profile(state.customer_id)
    calls += 1

    device_neighbors = None
    if txn.get("device_key"):
        device_neighbors = graph_client.get_device_neighbors(txn["device_key"])
        calls += 1

    region_cluster = None
    if txn.get("addr1"):
        center_ts = txn.get("ts")
        try:
            center_dt = datetime.fromisoformat(center_ts)
            frm = (center_dt - timedelta(days=3)).isoformat(sep=" ")
            to = (center_dt + timedelta(days=3)).isoformat(sep=" ")
            region_cluster = graph_client.get_region_cluster(txn["addr1"], frm, to)
            calls += 1
        except Exception:
            pass

    bundle = EvidenceBundle(
        txn=txn,
        card_window=card_window,
        card_region_history=card_region_history,
        customer_profile=customer_profile,
        device_neighbors=device_neighbors,
        region_cluster=region_cluster,
        tool_calls=calls,
    )
    state.evidence = bundle
    state.tool_calls += calls
    return bundle


# --- 2. Deterministic detection + probability aggregation ----------------


def run_detectors(state: InvestigationState) -> List[PatternSignal]:
    ev = state.evidence
    signals = run_all_detectors(ev.txn, ev.card_window, ev.card_region_history, ev.customer_profile)
    state.signals = signals
    return signals


def aggregate_base_probability(signals: List[PatternSignal], risk_score: Optional[float]) -> float:
    """Deterministic baseline: combine detector confidences (independent
    signals, combined as 1 - prod(1 - c_i), i.e. noisy-OR) with a modest
    pull from the model's risk_score, which the policy explicitly warns is
    "a reason to look, never a verdict" — so it's weighted lightly (15%)
    rather than trusted."""
    matched = [s for s in signals if s.matched]
    if not matched:
        base = 0.0
    else:
        prod = 1.0
        for s in matched:
            prod *= 1 - s.confidence
        base = 1 - prod

    if risk_score is not None:
        base = 0.85 * base + 0.15 * float(risk_score)

    return max(0.0, min(1.0, base))


# DeviceInfo/id_30/id_31 in this dataset are OS/browser/device-type strings
# ("Windows", "chrome 63.0", "iOS Device") — not unique per-user device IDs.
# Validated against the sampled dataset (docs/DECISIONS.md): median distinct
# cards per device_key is 1, but generic fingerprints are shared by up to
# 335 unrelated cards (e.g. bare "Windows", ~11% of all transactions).
# Treating every device_key match as fraud-ring evidence produced R6 false
# positives at scale (case HHG-017: 15+ "connected" cards from a device_key
# that was really just "Windows" desktop). A handful of cards sharing one
# device is plausible genuine signal (a reused/stolen device); a long tail
# is population noise from a common OS/browser string, not a fraud ring.
# Billing-region clusters get the same guard for the same reason.
MAX_MEANINGFUL_SHARED_CARDS = 5


def has_shared_origin(state: InvestigationState) -> Optional[str]:
    ev = state.evidence
    if ev.device_neighbors and ev.device_neighbors.get("cards"):
        n_cards = len(ev.device_neighbors["cards"])
        if 1 < n_cards <= MAX_MEANINGFUL_SHARED_CARDS:
            return f"device profile {ev.txn.get('device_key')}"
    if ev.region_cluster and ev.region_cluster.get("cards"):
        n_cards = len(ev.region_cluster["cards"])
        if 2 < n_cards <= MAX_MEANINGFUL_SHARED_CARDS:
            return f"billing region {ev.txn.get('addr1')}"
    return None


def connected_cards(state: InvestigationState) -> List[str]:
    """Cards to name as connected/monitored — only when the shared device or
    billing region is specific enough to be meaningful (see
    MAX_MEANINGFUL_SHARED_CARDS above). A device_key shared by dozens+ of
    cards is a generic OS/browser string, not a fraud ring, so it
    contributes no connected_card_ids at all.

    Must mirror has_shared_origin()'s two checks (device_neighbors AND
    region_cluster) exactly — this used to check device_neighbors only,
    so a region-only match (has_shared_origin returning "billing region
    ...", firing policy rule R6) produced connected_card_ids: [] even
    though R6's own reason text named a shared element. Found via a real
    HHG-001 run: R6 fired on "billing region 444.0" with zero connected
    cards reported."""
    ev = state.evidence
    cards = set()
    if ev.device_neighbors:
        dev_cards = ev.device_neighbors.get("cards", [])
        if 1 < len(dev_cards) <= MAX_MEANINGFUL_SHARED_CARDS:
            cards.update(dev_cards)
    if ev.region_cluster:
        region_cards = ev.region_cluster.get("cards", [])
        if 2 < len(region_cards) <= MAX_MEANINGFUL_SHARED_CARDS:
            cards.update(region_cards)
    cards.discard(state.card_id)
    return sorted(cards)


# --- 3. Case memory --------------------------------------------------------


def retrieve_memory(state: InvestigationState, graph_client, top_k: int = 5) -> List[Dict[str, Any]]:
    matched = [s for s in state.signals if s.matched]
    pattern_hint = matched[0].pattern if matched else "none"
    ev = state.evidence
    query_text = memory.case_narrative_text(
        pattern=pattern_hint,
        outcome_or_verdict="under investigation",
        exposure_usd=ev.txn.get("amt", 0.0),
        n_txns=len(ev.card_window.get("transactions", [])),
        analyst_notes=" ".join(c for s in matched for c in s.evidence_claims),
        channel=ev.txn.get("channel", ""),
    )
    results = memory.retrieve_similar_cases(graph_client, query_text, top_k=top_k)
    state.similar_cases = results
    state.tool_calls += 1
    return results


# --- 4. Evidence-request simulation ---------------------------------------


def decide_evidence_request(
    state: InvestigationState, uncertainty: UncertaintyAssessment
) -> Optional[EvidenceRequestType]:
    """Mirrors policy R1: a single weak signal below 0.70 should be
    verified before any block. If uncertainty is INSUFFICIENT and no
    customer reply has been simulated yet, request one."""
    if state.customer_response is not None:
        return None
    if state.step >= MAX_EVIDENCE_ROUNDS:
        return None
    if uncertainty.evidence_sufficiency == "INSUFFICIENT":
        if state.trigger_type == "customer_report":
            return EvidenceRequestType.customer_validation
        return EvidenceRequestType.customer_validation
    return None


def simulate_evidence_response(state: InvestigationState, req_type: EvidenceRequestType) -> str:
    """The README is explicit: customer/analyst replies are NOT provided —
    "simulate them in your own system and record what you assumed." This
    simulation is deterministic (not random) so runs are reproducible: it
    reads the same detector signals the rest of the pipeline used. A
    customer_report trigger where a real pattern matched is simulated as a
    denial (the customer already told us they don't recognize the charge);
    a risk-score/analyst trigger with strong pattern evidence is also
    simulated as a denial; weak/no evidence is simulated as a confirmation
    (nothing found beyond the score) — see docs/DECISIONS.md."""
    matched = [s for s in state.signals if s.matched]
    strong = any(s.confidence >= 0.5 for s in matched)

    if state.trigger_type == "customer_report":
        state.customer_response = "denied"
        assumed = "Customer states they did not make this purchase and still has the card (consistent with their original report)."
    elif strong:
        state.customer_response = "denied"
        assumed = "Customer states they did not recognize or make this transaction."
    elif matched:
        state.customer_response = "no_reply"
        assumed = "No reply received from the customer within 24 hours."
    else:
        state.customer_response = "confirmed"
        assumed = "Customer confirms they made this transaction."

    state.evidence_requests.append(
        EvidenceRequest(type=req_type, asked_after_step=state.step, assumed_response=assumed)
    )
    return state.customer_response


# --- 5. Policy decision -----------------------------------------------------


def build_policy_facts(
    state: InvestigationState,
    fraud_probability: float,
    uncertainty: UncertaintyAssessment,
    verdict: Verdict,
) -> policy.PolicyFacts:
    matched = [s for s in state.signals if s.matched]
    ev = state.evidence
    card_testing = any(s.pattern == "card_testing" and s.matched for s in state.signals)
    purchase_over_100 = ev.txn.get("amt", 0) > 100.0
    shared = has_shared_origin(state)
    conn_cards = connected_cards(state)

    return policy.PolicyFacts(
        fraud_probability=fraud_probability,
        pattern=(matched[0].pattern if matched else "none"),
        exposure_usd=sum(
            t["amt"]
            for t in ev.card_window.get("transactions", [])
            if t["txn_id"] in {tid for s in matched for tid in s.involved_txn_ids} | {state.flagged_txn_id}
        )
        or ev.txn.get("amt", 0.0),
        single_weak_signal=(len(matched) <= 1 and uncertainty.independent_evidence_count <= 1),
        card_testing_sequence=card_testing,
        purchase_over_100_cleared=card_testing and purchase_over_100,
        shared_origin=shared,
        connected_card_ids=conn_cards,
        customer_response=state.customer_response,
        verdict_uncertain=(verdict == Verdict.uncertain),
    )


# --- 6. LLM synthesis --------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are a fraud investigation analyst assistant at a bank. You are given
structured evidence gathered from a transaction graph and deterministic pattern-detector output for one
alert. Your job is ONLY to synthesize: choose the single best-fitting pattern (or 'undocumented' with a
description, or 'none' if evidence points to legitimate activity), write a short analyst summary, and note
anything the detectors might have missed in plain language. Do NOT invent transactions, IDs, or evidence
that isn't in the input. Do NOT decide next-best-actions — that comes from a separate deterministic policy
engine. Respond with STRICT JSON only, matching this shape:
{
  "pattern": "card_testing|card_not_present_fraud|card_not_present_new_device|out_of_region_use|account_takeover|undocumented|none",
  "pattern_description": "" or 2-3 sentences (required if pattern is 'undocumented'),
  "verdict": "fraud|legitimate|uncertain",
  "fraud_probability_adjustment": -0.15 to 0.15 (how much to nudge the deterministic baseline, and why in additional_notes),
  "summary": "2-6 sentences an analyst could read",
  "additional_evidence_claims": ["..."],
  "first_suspicious_txn_id": "..."
}
"""


def synthesize_with_llm(state: InvestigationState, llm, base_probability: float) -> Dict[str, Any]:
    ev = state.evidence
    matched = [s for s in state.signals if s.matched]
    payload = {
        "trigger_type": state.trigger_type,
        "trigger_text": state.trigger_text,
        "flagged_transaction": ev.txn,
        "detector_signals": [
            {
                "pattern": s.pattern,
                "confidence": s.confidence,
                "claims": s.evidence_claims,
                "txn_ids": s.involved_txn_ids,
            }
            for s in matched
        ],
        "customer_profile": {k: v for k, v in ev.customer_profile.items() if k != "cards"},
        "similar_prior_cases": state.similar_cases,
        "deterministic_base_probability": base_probability,
        "customer_response_so_far": state.customer_response,
    }
    text = llm.complete(
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        max_tokens=3000,
    )
    state.tokens += llm.usage.total_tokens
    return _parse_json_response(text)


SAR_SYSTEM_PROMPT = """You write Suspicious Activity Report narratives for a bank's regulatory filings,
following FinCEN SAR narrative guidance: who, what, when, where, how, and why it is suspicious. 6-12
sentences. Use ONLY the facts given below — no invented details. Respond with STRICT JSON:
{"narrative": "...", "subjects": ["..."]}
"""


def write_sar_narrative(state: InvestigationState, llm, case_fields: Dict[str, Any]) -> Dict[str, Any]:
    ev = state.evidence
    payload = {
        "customer_id": state.customer_id,
        "card_id": state.card_id,
        "pattern": case_fields.get("pattern"),
        "affected_txn_ids": case_fields.get("affected_txn_ids"),
        "exposure_usd": case_fields.get("exposure_usd"),
        "evidence": case_fields.get("evidence_claims"),
        "connected_card_ids": case_fields.get("connected_card_ids"),
        "customer_response": state.customer_response,
        "activity_dates": case_fields.get("activity_dates"),
    }
    text = llm.complete(
        system=SAR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        max_tokens=1500,
    )
    state.tokens += llm.usage.total_tokens
    return _parse_json_response(text)


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Providers vary in how they wrap structured output: Anthropic tends to
    return bare JSON when asked to; Gemini often wraps it in a ```json ...```
    fence regardless of the system prompt. Strip fences explicitly first
    (cheap, and makes the failure mode below unambiguous), THEN look for the
    {...} object.

    If no closing brace is found at all, the response was almost certainly
    truncated mid-JSON by the token budget (observed with Gemini's
    "thinking" flash models, which can spend part of max_output_tokens on
    internal reasoning before emitting the answer) rather than malformed —
    say so explicitly so it's not confused with a genuine formatting error.
    """
    text = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        if "{" in text and "}" not in text:
            raise ValueError(
                "LLM response was truncated before completing its JSON output "
                f"(no closing brace found) — increase max_tokens. Got: {text[:300]}"
            )
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(match.group(0))
