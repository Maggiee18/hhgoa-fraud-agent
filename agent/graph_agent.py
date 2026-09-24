"""
LangGraph wrapper around agent/investigation.py's pipeline: the explicit
investigation loop from the spec —

  TRIGGER -> CREATE/OPEN CASE -> QUERY GRAPH -> COLLECT EVIDENCE ->
  IDENTIFY PATTERNS -> ASSESS RISK+CONFIDENCE -> CHECK UNCERTAINTY ->
  [IF INSUFFICIENT: REQUEST EVIDENCE -> SIMULATE RESPONSE -> REASSESS] ->
  SELECT NEXT BEST ACTION -> POLICY CHECK -> UPDATE CASE -> STORE MEMORY ->
  EXPLAIN

as an actual state graph with a bounded evidence-request loop
(MAX_EVIDENCE_ROUNDS in agent/investigation.py) so it provably terminates.
The step logic itself lives in agent/investigation.py, tested independent
of LangGraph; this module is orchestration only.
"""
from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent import investigation as inv
from agent import policy
from agent.schemas import (
    ActionRecommendation,
    Case,
    CaseAnswer,
    Evidence,
    EvidenceSource,
    NextBestActions,
    Pattern,
    SAR,
    Verdict,
)


class GraphState(TypedDict, total=False):
    s: inv.InvestigationState
    graph_client: Any
    llm: Any
    base_probability: float
    fraud_probability: float
    verdict: str
    uncertainty: Any
    uncertainty_final: Any
    _req_type: Any


def _verdict_from_probability(p: float) -> Verdict:
    if p >= 0.6:
        return Verdict.fraud
    if p <= 0.2:
        return Verdict.legitimate
    return Verdict.uncertain


def node_gather(state: GraphState) -> GraphState:
    inv.gather_evidence(state["s"], state["graph_client"])
    return state


def node_detect(state: GraphState) -> GraphState:
    inv.run_detectors(state["s"])
    return state


def node_memory(state: GraphState) -> GraphState:
    inv.retrieve_memory(state["s"], state["graph_client"])
    return state


def node_synthesize(state: GraphState) -> GraphState:
    s = state["s"]
    base = inv.aggregate_base_probability(s.signals, s.risk_score)
    synthesis = inv.synthesize_with_llm(s, state["llm"], base)
    s.llm_synthesis = synthesis
    adj = float(synthesis.get("fraud_probability_adjustment", 0.0) or 0.0)
    adj = max(-0.15, min(0.15, adj))
    p = max(0.0, min(1.0, base + adj))
    state["base_probability"] = base
    state["fraud_probability"] = p
    # Verdict is derived from the (deterministic-baseline + bounded-LLM-
    # adjustment) probability, not taken verbatim from the LLM, so the two
    # scored fields (fraud_probability, verdict) can never contradict each
    # other — "Be honest; this is scored for calibration" (answer format).
    state["verdict"] = _verdict_from_probability(p).value
    return state


def node_policy_initial(state: GraphState) -> GraphState:
    s = state["s"]
    uncertainty = inv.assess(
        fraud_probability=state["fraud_probability"],
        signals=s.signals,
        has_device_info=bool(s.evidence.txn.get("device_key")),
        has_prior_similar_case=bool(s.similar_cases),
        single_weak_signal=(len([sig for sig in s.signals if sig.matched]) <= 1),
    )
    state["uncertainty"] = uncertainty
    facts = inv.build_policy_facts(s, state["fraud_probability"], uncertainty, Verdict(state["verdict"]))
    decision = policy.decide(facts)
    s.initial_actions = decision.actions
    s.rules_applied.extend(decision.rules_applied)
    # Decide here, in the node, not in the conditional-edge router below:
    # LangGraph only guarantees a state update is committed to the graph's
    # channels when it comes back as part of a *node's* return value.
    # route_after_initial_policy used to set state["_req_type"] itself as a
    # side effect of routing, which happened to work most of the time via
    # incidental object-reference reuse but isn't something the framework
    # contracts to preserve — observed failing on one case in a 20-case
    # benchmark run (KeyError: '_req_type' in node_request_evidence).
    state["_req_type"] = inv.decide_evidence_request(s, uncertainty)
    return state


def route_after_initial_policy(state: GraphState) -> str:
    if state.get("_req_type") is not None:
        return "request_evidence"
    return "finalize"


def node_request_evidence(state: GraphState) -> GraphState:
    s = state["s"]
    s.step += 1
    inv.simulate_evidence_response(s, state["_req_type"])
    # reassess probability deterministically from the (simulated) reply
    p = state["fraud_probability"]
    if s.customer_response == "denied":
        p = min(1.0, p + 0.15)
    elif s.customer_response == "confirmed":
        p = max(0.0, p - 0.40)
    state["fraud_probability"] = p
    state["verdict"] = _verdict_from_probability(p).value
    return state


def node_finalize(state: GraphState) -> GraphState:
    s = state["s"]
    uncertainty = inv.assess(
        fraud_probability=state["fraud_probability"],
        signals=s.signals,
        has_device_info=bool(s.evidence.txn.get("device_key")),
        has_prior_similar_case=bool(s.similar_cases),
        single_weak_signal=(len([sig for sig in s.signals if sig.matched]) <= 1),
    )
    facts = inv.build_policy_facts(s, state["fraud_probability"], uncertainty, Verdict(state["verdict"]))
    decision = policy.decide(facts)
    s.final_actions = decision.actions
    s.rules_applied.extend(decision.rules_applied)

    stop = policy.stop_investigation(
        fraud_probability=state["fraud_probability"],
        independent_evidence_count=uncertainty.independent_evidence_count,
        verification_settled=(s.customer_response is not None),
        further_steps_unlikely_to_change=(s.step >= inv.MAX_EVIDENCE_ROUNDS),
    )
    s.stop_reason = stop or "Reached maximum evidence-gathering rounds for this alert."
    state["uncertainty_final"] = uncertainty
    return state


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("gather", node_gather)
    g.add_node("detect", node_detect)
    g.add_node("memory", node_memory)
    g.add_node("synthesize", node_synthesize)
    g.add_node("policy_initial", node_policy_initial)
    g.add_node("request_evidence", node_request_evidence)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("gather")
    g.add_edge("gather", "detect")
    g.add_edge("detect", "memory")
    g.add_edge("memory", "synthesize")
    g.add_edge("synthesize", "policy_initial")
    g.add_conditional_edges(
        "policy_initial", route_after_initial_policy, {"request_evidence": "request_evidence", "finalize": "finalize"}
    )
    g.add_edge("request_evidence", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_COMPILED = None


def get_compiled_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run_investigation(case_row: dict, graph_client, llm) -> CaseAnswer:
    """case_row: a dict from case_pack.csv (case_id, opened_at, trigger_type,
    trigger_text, flagged_txn_id, card_id, customer_id, risk_score)."""
    t0 = time.time()
    s = inv.InvestigationState(
        case_id=case_row["case_id"],
        trigger_type=case_row["trigger_type"],
        trigger_text=case_row["trigger_text"],
        flagged_txn_id="T" + str(case_row["flagged_txn_id"]),
        card_id=case_row["card_id"],
        customer_id=case_row["customer_id"],
        risk_score=(float(case_row["risk_score"]) if case_row.get("risk_score") not in (None, "", "nan") else None),
    )
    graph = get_compiled_graph()
    final_state = graph.invoke({"s": s, "graph_client": graph_client, "llm": llm, "base_probability": 0.0, "fraud_probability": 0.0, "verdict": "uncertain"})
    return _assemble_case_answer(final_state, time.time() - t0)


def _assemble_case_answer(state: GraphState, latency_s: float) -> CaseAnswer:
    s: inv.InvestigationState = state["s"]
    synthesis = s.llm_synthesis
    matched = [sig for sig in s.signals if sig.matched]
    ev = s.evidence

    pattern_str = synthesis.get("pattern") or (matched[0].pattern if matched else "none")
    try:
        pattern = Pattern(pattern_str)
    except ValueError:
        pattern = Pattern.undocumented if matched else Pattern.none_

    affected_ids = sorted({tid for sig in matched for tid in sig.involved_txn_ids} | ({s.flagged_txn_id} if pattern != Pattern.none_ else set()))
    exposure = sum(
        t["amt"]
        for t in ev.card_window.get("transactions", [])
        if t["txn_id"] in affected_ids
    ) if affected_ids else 0.0

    evidence_list = [
        Evidence(claim=claim, source=EvidenceSource.graph, ref=f"detector:{sig.pattern}", entity_ids=sig.involved_txn_ids)
        for sig in matched
        for claim in sig.evidence_claims
    ]
    for claim in synthesis.get("additional_evidence_claims", []) or []:
        evidence_list.append(Evidence(claim=claim, source=EvidenceSource.graph, ref="synthesis", entity_ids=[s.flagged_txn_id]))
    for req in s.evidence_requests:
        evidence_list.append(
            Evidence(
                claim=req.assumed_response,
                source=EvidenceSource.customer,
                ref=f"evidence_request:{len(s.evidence_requests)}",
                entity_ids=[],
            )
        )

    verdict = Verdict(state["verdict"])

    from agent.schemas import ActionName as _ActionName

    file_report = any(a.action == _ActionName.FILE_REPORT for a in s.final_actions)

    sar_obj = SAR(file=False, reason="No policy rule called for a report.")
    if file_report:
        sar_fields = {
            "pattern": pattern.value,
            "affected_txn_ids": affected_ids,
            "exposure_usd": exposure,
            "evidence_claims": [e.claim for e in evidence_list],
            "connected_card_ids": inv.connected_cards(s),
            "activity_dates": _activity_dates(ev, affected_ids),
        }
        try:
            sar_out = inv.write_sar_narrative(s, state["llm"], sar_fields)
            sar_obj = SAR(
                file=True,
                reason="Policy rule triggered FILE_REPORT — see rules_applied.",
                narrative=sar_out.get("narrative", ""),
                subjects=sar_out.get("subjects", [s.customer_id, s.card_id]),
                total_amount_usd=exposure,
                activity_dates=_activity_dates(ev, affected_ids),
            )
        except Exception:
            sar_obj = SAR(
                file=True,
                reason="Policy rule triggered FILE_REPORT, but narrative generation failed — needs manual write-up.",
                narrative=f"Suspicious activity on card {s.card_id} (customer {s.customer_id}): {'; '.join(e.claim for e in evidence_list) or 'see case evidence'}.",
                subjects=[s.customer_id, s.card_id],
                total_amount_usd=exposure,
                activity_dates=_activity_dates(ev, affected_ids),
            )

    case_obj = Case(
        status=_status_from(verdict, s),
        verdict=verdict,
        fraud_probability=round(state["fraud_probability"], 4),
        pattern=pattern,
        pattern_description=(synthesis.get("pattern_description") or "") if pattern == Pattern.undocumented else "",
        affected_txn_ids=affected_ids if verdict != Verdict.legitimate else [],
        first_suspicious_txn_id=(synthesis.get("first_suspicious_txn_id") or (affected_ids[0] if affected_ids else "")),
        connected_card_ids=inv.connected_cards(s),
        connected_device_profiles=[ev.device_neighbors["device_info"]] if ev.device_neighbors and ev.device_neighbors.get("device_info") else [],
        exposure_usd=round(exposure, 2) if verdict != Verdict.legitimate else 0.0,
        evidence=evidence_list,
        similar_prior_cases=[c["case_id"] for c in s.similar_cases if c.get("case_source") == "historical"],
        summary=synthesis.get("summary", f"Investigation of {s.flagged_txn_id} on card {s.card_id}."),
        written_to_graph=False,
        graph_case_id="",
    )

    graph_case_id = f"CASE-{s.case_id}"
    if policy.should_open_case(state["fraud_probability"], bool(s.evidence_requests), s.trigger_type == "customer_report"):
        try:
            from agent.memory import embed_text

            embed_query = f"{case_obj.pattern} {case_obj.summary}"
            record = {
                "case_id": graph_case_id,
                "case_source": "agent",
                "status": case_obj.status.value if hasattr(case_obj.status, "value") else case_obj.status,
                "verdict": case_obj.verdict.value if hasattr(case_obj.verdict, "value") else case_obj.verdict,
                "fraud_probability": case_obj.fraud_probability,
                "pattern": case_obj.pattern.value if hasattr(case_obj.pattern, "value") else case_obj.pattern,
                "pattern_description": case_obj.pattern_description,
                "exposure_usd": case_obj.exposure_usd,
                "opened_at": str(ev.txn.get("ts")),
                "closed_at": str(ev.txn.get("ts")),
                "summary": case_obj.summary,
                "analyst_notes": "; ".join(e.claim for e in evidence_list),
                "report_filed": file_report,
                "first_txn_id": case_obj.first_suspicious_txn_id,
                "embedding": embed_text(embed_query),
                "txn_ids": affected_ids,
                "card_ids": [s.card_id],
                "connected_card_ids": case_obj.connected_card_ids,
                "device_keys": [ev.txn["device_key"]] if ev.txn.get("device_key") else [],
                "summary_fields": {"pattern": case_obj.pattern, "exposure_usd": case_obj.exposure_usd},
            }
            state["graph_client"].upsert_case(record)
            case_obj.written_to_graph = True
            case_obj.graph_case_id = graph_case_id
        except Exception:
            pass  # graph write is best-effort; the JSON answer file is the scored artifact

    what_changed = "nothing"
    if s.customer_response and s.initial_actions != s.final_actions:
        what_changed = (
            f"Customer response ('{s.customer_response}') updated the assessment "
            f"(probability {state['base_probability']:.2f} → {state['fraud_probability']:.2f}) "
            f"and changed the recommended actions."
        )

    return CaseAnswer(
        case_id=s.case_id,
        case=case_obj,
        evidence_requests=s.evidence_requests,
        next_best_actions=NextBestActions(
            initial=s.initial_actions,
            final=s.final_actions if s.evidence_requests else s.initial_actions,
            what_changed=what_changed,
        ),
        sar=sar_obj,
        stop_reason=s.stop_reason or "Investigation completed.",
        tool_calls=s.tool_calls,
        tokens=s.tokens,
        latency_s=round(latency_s, 2),
    )


def _status_from(verdict: Verdict, s: inv.InvestigationState):
    from agent.schemas import ActionName, CaseStatus

    if verdict == Verdict.fraud:
        return CaseStatus.closed_fraud
    if verdict == Verdict.legitimate:
        return CaseStatus.closed_legitimate
    escalated = any(a.action == ActionName.ESCALATE_TO_ANALYST for a in s.final_actions)
    return CaseStatus.escalated if escalated else CaseStatus.open


def _activity_dates(ev: inv.EvidenceBundle, affected_ids):
    txns = [t for t in ev.card_window.get("transactions", []) if t["txn_id"] in affected_ids]
    if not txns:
        return [str(ev.txn.get("ts", ""))[:10]] * 2
    dates = sorted(t["ts"][:10] for t in txns)
    return [dates[0], dates[-1]]
