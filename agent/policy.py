"""
Deterministic fraud policy engine — Fraud Policy v1.0 from data/README.md.

Kept separate from the LLM reasoning layer on purpose (see the hackathon
spec: "prefer deterministic components for fraud scoring/policy/action
selection"). The LLM proposes evidence-backed facts (probability, pattern,
shared-origin findings, customer reply); this module turns those facts into
the exact action list + approval route + rule citation. This makes action
selection reproducible and auditable, and keeps the LLM from inventing
actions or approval routes that don't exist in the policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agent.schemas import ActionName, ApprovalRoute, ActionRecommendation

# --- 2. Approval routing -----------------------------------------------

AUTO_ACTIONS = {
    ActionName.ALLOW_TRANSACTION,
    ActionName.MONITOR_CARD,
    ActionName.MONITOR_CONNECTED_CARDS,
    ActionName.WARN_CUSTOMER,
    ActionName.VERIFY_WITH_CUSTOMER,
    ActionName.STEP_UP_AUTH,
    ActionName.GENERATE_REPORT,
    ActionName.CREATE_CASE,
    ActionName.ESCALATE_TO_ANALYST,
    ActionName.CLOSE_NO_FRAUD,
}


def route_for(action: ActionName, exposure_usd: float = 0.0) -> ApprovalRoute:
    if action in AUTO_ACTIONS:
        return ApprovalRoute.auto
    if action == ActionName.DECLINE_TRANSACTION:
        return ApprovalRoute.L1
    if action == ActionName.BLOCK_CARD:
        return ApprovalRoute.L1 if exposure_usd <= 2500 else ApprovalRoute.L2
    if action == ActionName.BLOCK_ALL_CARDS:
        return ApprovalRoute.L2
    if action == ActionName.FILE_REPORT:
        return ApprovalRoute.L2
    raise ValueError(f"Unknown action for routing: {action}")


# --- Policy input / output shapes ---------------------------------------


@dataclass
class PolicyFacts:
    """What the investigation has established, going into a policy decision."""

    fraud_probability: float
    pattern: str  # Pattern enum value, or "none"/"undocumented"
    exposure_usd: float
    single_weak_signal: bool  # True if resting on one signal (incl. risk score alone)
    card_testing_sequence: bool = False
    purchase_over_100_cleared: bool = False  # for R5's second half
    shared_origin: Optional[str] = None  # e.g. "device profile D000731" — set => R6
    connected_card_ids: List[str] = field(default_factory=list)
    customer_response: Optional[str] = None  # None | "denied" | "confirmed" | "no_reply"
    disputed_but_recurring: bool = False  # R7
    coordinated_undocumented: bool = False  # R9
    verdict_uncertain: bool = False  # for R8


@dataclass
class PolicyDecision:
    actions: List[ActionRecommendation]
    rules_applied: List[str]


def _rec(action: ActionName, reason: str, exposure_usd: float = 0.0) -> ActionRecommendation:
    return ActionRecommendation(action=action, route=route_for(action, exposure_usd), reason=reason)


def decide(facts: PolicyFacts) -> PolicyDecision:
    """
    Apply rules R1-R10 (+ 3a case/report logic) in the order the policy
    implies: response-driven rules (R2/R3/R4) first when a customer has
    replied, then pattern rules (R5/R6/R7/R9), then the general weak-signal
    rule (R1) and escalation (R8), each only firing if its precondition
    holds. `rules_applied` records exactly which fired, for the
    explanation ("cite the rule number", README section 7).
    """
    actions: List[ActionRecommendation] = []
    rules: List[str] = []
    exposure = facts.exposure_usd

    # R3 — customer confirms: closes the loop, nothing else matters.
    if facts.customer_response == "confirmed":
        actions.append(_rec(ActionName.CLOSE_NO_FRAUD, "R3: customer confirmed the transaction"))
        rules.append("R3")
        return PolicyDecision(actions=_dedupe(actions), rules_applied=rules)

    # R2 — customer denies.
    if facts.customer_response == "denied":
        actions.append(_rec(ActionName.BLOCK_CARD, "R2: customer denied the transaction", exposure))
        actions.append(_rec(ActionName.CREATE_CASE, "R2"))
        rules.append("R2")
        if exposure > 1000 or facts.shared_origin:
            reason = "R2: exposure exceeds $1,000" if exposure > 1000 else f"R2: connects to shared origin ({facts.shared_origin})"
            actions.append(_rec(ActionName.FILE_REPORT, reason))
        if facts.connected_card_ids:
            actions.append(_rec(ActionName.MONITOR_CONNECTED_CARDS, f"Shared origin with {', '.join(facts.connected_card_ids)}"))

    # R4 — no reply within 24h.
    if facts.customer_response == "no_reply":
        actions.append(_rec(ActionName.MONITOR_CARD, "R4: no reply within 24 hours"))
        actions.append(_rec(ActionName.DECLINE_TRANSACTION, "R4: no reply, pending authorization", exposure))
        rules.append("R4")
        if exposure > 500:
            actions.append(_rec(ActionName.ESCALATE_TO_ANALYST, "R4: exposure exceeds $500"))

    # R7 — disputed but matches recurring pattern.
    if facts.disputed_but_recurring:
        actions.append(_rec(ActionName.CREATE_CASE, "R7: disputed charge matches cardholder's own recurring pattern"))
        actions.append(_rec(ActionName.VERIFY_WITH_CUSTOMER, "R7"))
        actions.append(_rec(ActionName.WARN_CUSTOMER, "R7"))
        rules.append("R7")
        return PolicyDecision(actions=_dedupe(actions), rules_applied=rules)  # R7 explicitly: do not block

    # R5 — card testing.
    if facts.card_testing_sequence:
        actions.append(_rec(ActionName.DECLINE_TRANSACTION, "R5: card testing sequence observed", exposure))
        actions.append(_rec(ActionName.STEP_UP_AUTH, "R5"))
        rules.append("R5")
        if facts.purchase_over_100_cleared:
            actions.append(_rec(ActionName.BLOCK_CARD, "R5: purchase over $100 already cleared", exposure))

    # R6 — shared origin across cards.
    if facts.shared_origin and facts.customer_response != "denied":
        actions.append(_rec(ActionName.CREATE_CASE, f"R6: shared origin ({facts.shared_origin}) across cards"))
        actions.append(_rec(ActionName.FILE_REPORT, f"R6: named shared element — {facts.shared_origin}"))
        if facts.connected_card_ids:
            actions.append(_rec(ActionName.MONITOR_CONNECTED_CARDS, f"R6: cards sharing {facts.shared_origin}"))
        rules.append("R6")

    # R9 — undocumented but coordinated/repeated abuse.
    if facts.coordinated_undocumented:
        actions.append(_rec(ActionName.CREATE_CASE, "R9: undocumented pattern, coordinated/repeated abuse"))
        actions.append(_rec(ActionName.FILE_REPORT, "R9"))
        actions.append(_rec(ActionName.ESCALATE_TO_ANALYST, "R9"))
        rules.append("R9")

    # R1 — weak single signal, not yet resolved by a customer reply.
    if facts.customer_response is None and facts.single_weak_signal and facts.fraud_probability < 0.70:
        actions.append(_rec(ActionName.VERIFY_WITH_CUSTOMER, f"R1: probability {facts.fraud_probability:.2f} on a single signal"))
        rules.append("R1")

    # R8 — escalate when uncertain and exposed.
    if facts.verdict_uncertain and (exposure > 500):
        actions.append(_rec(ActionName.ESCALATE_TO_ANALYST, "R8: verdict uncertain and exposure exceeds $500"))
        rules.append("R8")

    # R10 guard is enforced by the caller (never propose BLOCK_ALL_CARDS
    # here unless the caller's facts explicitly justify it) — this engine
    # simply never emits BLOCK_ALL_CARDS on its own; callers that need it
    # append it explicitly after checking R10's condition themselves.

    if not actions:
        # Nothing triggered — fall back to a defensible default so the
        # investigation always produces *some* recommendation.
        if facts.fraud_probability <= 0.15:
            actions.append(_rec(ActionName.CLOSE_NO_FRAUD, "No rule triggered; fraud probability at/below 0.15"))
            rules.append("default-clear")
        else:
            actions.append(_rec(ActionName.GENERATE_REPORT, "No rule triggered; recording the review"))
            rules.append("default-record")

    return PolicyDecision(actions=_dedupe(actions), rules_applied=rules)


def should_file_report(final_actions: List[ActionRecommendation]) -> bool:
    return any(a.action == ActionName.FILE_REPORT for a in final_actions)


def should_open_case(fraud_probability: float, evidence_requested: bool, customer_disputed: bool) -> bool:
    """3a: open a case whenever fraud probability reaches 0.30, whenever
    evidence is requested, or whenever a customer disputes a charge."""
    return fraud_probability >= 0.30 or evidence_requested or customer_disputed


def stop_investigation(
    fraud_probability: float,
    independent_evidence_count: int,
    verification_settled: bool,
    further_steps_unlikely_to_change: bool,
) -> Optional[str]:
    """Section 6 stopping rule. Returns a stop_reason string once one of
    the conditions holds, else None (keep investigating)."""
    if (fraud_probability >= 0.85 or fraud_probability <= 0.15) and independent_evidence_count >= 2:
        return (
            f"Fraud probability {fraud_probability:.2f} is decisive and supported by "
            f"{independent_evidence_count} independent pieces of evidence."
        )
    if verification_settled:
        return "A verification response settled the question."
    if further_steps_unlikely_to_change:
        return "Further steps are unlikely to change the decision."
    return None


def _dedupe(actions: List[ActionRecommendation]) -> List[ActionRecommendation]:
    seen = set()
    out = []
    for a in actions:
        if a.action not in seen:
            seen.add(a.action)
            out.append(a)
    return out
