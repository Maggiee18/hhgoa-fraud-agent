from tests.fakes import FakeGraphClient, FakeLLM
from agent.graph_agent import run_investigation
from agent.schemas import ActionName


def test_no_signal_low_risk_score_closes_clean():
    """Half the cases are legitimate per the README — an agent that blocks
    everything scores badly. This is the "nothing here" path: no detector
    fires, risk score is low, single weak signal -> R1 verify, and after a
    simulated confirmation, CLOSE_NO_FRAUD."""
    card_id = "C99999-K1"
    customer_id = "C99999"
    flagged = "T9999001"
    txns = {
        flagged: {
            "found": True, "txn_id": flagged, "card_id": card_id, "customer_id": customer_id,
            "ts": "2016-12-01 10:00:00", "amt": 42.00, "product_cd": "W", "channel": "in_person",
            "risk_score": 0.2, "addr1": "100", "device_key": None, "device_new": None, "proxy": None,
            "c_features": {}, "d_features": {}, "m_features": {}, "v_features_nonnull_count": 0,
        }
    }
    card_window = [
        {"txn_id": "T9999000", "ts": "2016-11-01 09:00:00", "amt": 38.0, "product_cd": "W", "channel": "in_person", "risk_score": 0.1, "addr1": "100", "device_key": None},
        {"txn_id": flagged, "ts": "2016-12-01 10:00:00", "amt": 42.0, "product_cd": "W", "channel": "in_person", "risk_score": 0.2, "addr1": "100", "device_key": None},
    ]
    profiles = {customer_id: {"customer_id": customer_id, "found": True, "cards": [card_id], "txn_count": 2, "total_amt": 80.0, "channels_used": ["in_person"], "prior_closed_cases": []}}
    region_histories = {card_id: {"card_id": card_id, "regions": {"100": 2}}}

    client = FakeGraphClient(txns, {card_id: card_window}, profiles, region_histories=region_histories, similar_cases=[])
    llm = FakeLLM(
        synthesis_response={
            "pattern": "none", "pattern_description": "", "verdict": "legitimate",
            "fraud_probability_adjustment": 0.0, "summary": "Routine in-person purchase consistent with history.",
            "additional_evidence_claims": [], "first_suspicious_txn_id": "",
        },
        sar_response={"narrative": "", "subjects": []},
    )
    case_row = {
        "case_id": "HHG-TEST-LEGIT", "trigger_type": "risk_score",
        "trigger_text": "Real-time model scored transaction at 0.2.",
        "flagged_txn_id": "9999001", "card_id": card_id, "customer_id": customer_id, "risk_score": 0.2,
    }
    answer = run_investigation(case_row, client, llm)

    assert answer.case.pattern == "none"
    assert answer.case.affected_txn_ids == []
    assert answer.case.exposure_usd == 0.0
    assert answer.sar.file is False
    final_actions = {a.action for a in answer.next_best_actions.final}
    assert ActionName.BLOCK_CARD not in final_actions
    assert ActionName.CLOSE_NO_FRAUD in final_actions or ActionName.VERIFY_WITH_CUSTOMER in final_actions
