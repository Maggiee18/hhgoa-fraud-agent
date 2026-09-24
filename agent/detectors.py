"""
Deterministic pattern detectors — pure functions over graph tool output.

Per the engineering rules ("prefer deterministic components for fraud
scoring/policy/action selection"), pattern matching against the five known
fraud patterns is computed here, not guessed by the LLM. Each detector
returns a PatternSignal with a confidence in [0, 1] and the concrete
evidence it found; agent/investigation.py aggregates these into a base
fraud probability, and the LLM (agent/llm.py) is used only to turn the
signals into readable evidence claims / narrative and to weigh genuinely
ambiguous cases the detectors can't resolve — never to invent whether a
sequence of transactions happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PatternSignal:
    pattern: str
    matched: bool
    confidence: float  # 0..1, this detector's own confidence
    evidence_claims: List[str] = field(default_factory=list)
    involved_txn_ids: List[str] = field(default_factory=list)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def detect_card_testing(card_window: Dict[str, Any], flagged_txn_id: str) -> PatternSignal:
    """Pattern 1 / R5: 3+ small (<$5) online authorizations within an hour,
    then a larger purchase."""
    txns = sorted(card_window.get("transactions", []), key=lambda t: t["ts"])
    if len(txns) < 4:
        return PatternSignal("card_testing", False, 0.0)

    for i in range(len(txns) - 3):
        window = txns[i : i + 4]
        small = [t for t in window[:-1] if t["amt"] < 5.0 and t.get("channel") == "online"]
        last = window[-1]
        if len(small) >= 3:
            t0 = _parse_ts(small[0]["ts"])
            t_last_small = _parse_ts(small[-1]["ts"])
            if (t_last_small - t0).total_seconds() <= 3600 and last["amt"] > 100.0:
                return PatternSignal(
                    pattern="card_testing",
                    matched=True,
                    confidence=0.75,
                    evidence_claims=[
                        f"{len(small)} online authorizations under $5 within "
                        f"{int((t_last_small - t0).total_seconds() / 60)} minutes, "
                        f"followed by a ${last['amt']:.2f} purchase"
                    ],
                    involved_txn_ids=[t["txn_id"] for t in small] + [last["txn_id"]],
                )
    return PatternSignal("card_testing", False, 0.0)


def detect_cnp_fraud(
    txn: Dict[str, Any], card_window: Dict[str, Any], card_region_history: Dict[str, Any]
) -> PatternSignal:
    """Pattern 2: online use, amount/product inconsistent with history, often
    a burst of 2-4 within 48h. Ambiguous alone."""
    if txn.get("channel") != "online":
        return PatternSignal("card_not_present_fraud", False, 0.0)

    txns = sorted(card_window.get("transactions", []), key=lambda t: t["ts"])
    flagged_ts = _parse_ts(txn["ts"])
    burst = [
        t for t in txns if abs((_parse_ts(t["ts"]) - flagged_ts).total_seconds()) <= 48 * 3600
    ]
    online_burst = [t for t in burst if t.get("channel") == "online"]

    hist_amts = [t["amt"] for t in txns if t["txn_id"] != txn["txn_id"]]
    avg_amt = sum(hist_amts) / len(hist_amts) if hist_amts else txn["amt"]
    amt_ratio = txn["amt"] / avg_amt if avg_amt > 0 else 1.0

    matched = len(online_burst) >= 2 and (amt_ratio > 2.5 or amt_ratio < 0.2)
    confidence = 0.35 if matched else 0.0
    claims = []
    if matched:
        claims.append(
            f"{len(online_burst)} online transactions within 48h of the flagged one; "
            f"amount ${txn['amt']:.2f} is {amt_ratio:.1f}x this card's average (${avg_amt:.2f})"
        )
    return PatternSignal(
        "card_not_present_fraud",
        matched,
        confidence,
        claims,
        [t["txn_id"] for t in online_burst] if matched else [],
    )


def detect_cnp_new_device(txn_details: Dict[str, Any]) -> PatternSignal:
    """Pattern 3: same as CNP fraud, but identity record marks device New,
    sometimes behind a proxy. Stronger than pattern 2 alone."""
    if txn_details.get("channel") != "online":
        return PatternSignal("card_not_present_new_device", False, 0.0)
    device_new = txn_details.get("device_new")
    proxy = txn_details.get("proxy")
    if device_new == "New":
        confidence = 0.55
        claims = [f"Transaction from a device profile marked New for this account (proxy: {proxy or 'not flagged'})"]
        if proxy and str(proxy).lower() not in ("transparent", "", "none"):
            confidence = 0.65
            claims[0] += " — proxy rating is elevated"
        return PatternSignal(
            "card_not_present_new_device", True, confidence, claims, [txn_details["txn_id"]]
        )
    return PatternSignal("card_not_present_new_device", False, 0.0)


def detect_out_of_region(
    txn: Dict[str, Any], card_region_history: Dict[str, Any]
) -> PatternSignal:
    """Pattern 4: card-present purchase in a region the card has no history
    in, while normal activity continues at home. A single new region with
    ongoing history elsewhere and few txns there suggests fraud; several
    days of activity in the new region suggests a trip (not fraud) —
    detector flags the signal, agent/investigation.py's narrative should
    say which it looks like based on txn count in that region."""
    if txn.get("channel") == "online":
        return PatternSignal("out_of_region_use", False, 0.0)
    region = txn.get("addr1")
    regions = card_region_history.get("regions", {})
    if region is None:
        return PatternSignal("out_of_region_use", False, 0.0)
    region_key = str(region)
    count_here = regions.get(region_key, 0)
    total = sum(regions.values()) or 1
    is_new_or_rare = count_here <= 2 and total > count_here
    if is_new_or_rare:
        confidence = 0.45 if count_here <= 1 else 0.30
        return PatternSignal(
            "out_of_region_use",
            True,
            confidence,
            [
                f"Card-present purchase in billing region {region_key}, "
                f"which this card has {count_here} prior transaction(s) in out of {total} total"
            ],
            [txn["txn_id"]],
        )
    return PatternSignal("out_of_region_use", False, 0.0)


def detect_account_takeover(
    txn_details: Dict[str, Any], customer_profile: Dict[str, Any], card_window: Dict[str, Any]
) -> PatternSignal:
    """Pattern 5: mixed-channel activity inconsistent with the cardholder,
    device/match-flag anomalies. Weakest-evidenced detector — mostly flags
    for the LLM to weigh alongside everything else, since 'inconsistent
    with the cardholder' is inherently judgment-based."""
    channels = set(customer_profile.get("channels_used", []))
    m_features = txn_details.get("m_features", {})
    mismatch_flags = [k for k, v in m_features.items() if v == "F" or v == 0]
    match_status = txn_details.get("match_status")
    signals = 0
    claims = []
    if len(channels) > 1:
        signals += 1
        claims.append(f"Customer's history spans both channels: {sorted(channels)}")
    if mismatch_flags:
        signals += 1
        claims.append(f"{len(mismatch_flags)} match-flag(s) indicate mismatch (e.g. name/address)")
    if match_status and str(match_status) not in ("M", "1", "match"):
        signals += 1
        claims.append(f"Identity match status: {match_status}")
    matched = signals >= 2
    return PatternSignal(
        "account_takeover",
        matched,
        0.4 if matched else 0.0,
        claims if matched else [],
        [txn_details["txn_id"]] if matched else [],
    )


def run_all_detectors(
    txn_details: Dict[str, Any],
    card_window: Dict[str, Any],
    card_region_history: Dict[str, Any],
    customer_profile: Dict[str, Any],
) -> List[PatternSignal]:
    flagged_id = txn_details["txn_id"]
    return [
        detect_card_testing(card_window, flagged_id),
        detect_cnp_fraud(txn_details, card_window, card_region_history),
        detect_cnp_new_device(txn_details),
        detect_out_of_region(txn_details, card_region_history),
        detect_account_takeover(txn_details, customer_profile, card_window),
    ]
