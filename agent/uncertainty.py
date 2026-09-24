"""
Structured uncertainty tracking, kept separate from the narrative summary
so the dashboard can render it directly (risk / confidence / evidence
sufficiency / unresolved questions / conflicting evidence / missing
evidence — the "Uncertainty Panel" in the spec) and so
agent/policy.stop_investigation gets clean, structured inputs instead of
parsing prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agent.detectors import PatternSignal


@dataclass
class UncertaintyAssessment:
    risk: str  # LOW | MEDIUM | HIGH
    fraud_probability: float
    evidence_sufficiency: str  # SUFFICIENT | INSUFFICIENT
    independent_evidence_count: int
    uncertainty_reasons: List[str] = field(default_factory=list)
    conflicting_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)


def _risk_band(p: float) -> str:
    if p >= 0.7:
        return "HIGH"
    if p >= 0.35:
        return "MEDIUM"
    return "LOW"


def assess(
    fraud_probability: float,
    signals: List[PatternSignal],
    has_device_info: bool,
    has_prior_similar_case: bool,
    single_weak_signal: bool,
) -> UncertaintyAssessment:
    matched = [s for s in signals if s.matched]
    independent_evidence_count = len(matched) + (1 if has_prior_similar_case else 0)

    reasons: List[str] = []
    missing: List[str] = []
    conflicting: List[str] = []

    if not has_device_info:
        missing.append("No identity/device record for this transaction (in-person or unmatched)")
    if len(matched) == 0:
        reasons.append("No pattern detector fired — assessment rests on the risk score / trigger alone")
    if len(matched) > 1:
        pats = {s.pattern for s in matched}
        if len(pats) > 1:
            conflicting.append(f"Multiple distinct patterns triggered: {sorted(pats)} — needs a call on which dominates")
    if single_weak_signal:
        reasons.append("Case rests on a single signal per policy R1")
    if not has_prior_similar_case:
        missing.append("No similar prior case found in case memory")

    sufficiency = "SUFFICIENT" if independent_evidence_count >= 2 or fraud_probability <= 0.15 or fraud_probability >= 0.85 else "INSUFFICIENT"

    return UncertaintyAssessment(
        risk=_risk_band(fraud_probability),
        fraud_probability=fraud_probability,
        evidence_sufficiency=sufficiency,
        independent_evidence_count=independent_evidence_count,
        uncertainty_reasons=reasons,
        conflicting_evidence=conflicting,
        missing_evidence=missing,
    )
