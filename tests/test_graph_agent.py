from tests.fakes import FakeGraphClient, FakeLLM
from agent.graph_agent import run_investigation
from agent.schemas import ActionName


def _card_testing_fixture():
    flagged = "T3450629"
    card_id = "C04570-K1"
    customer_id = "C04570"

    txns = {
        flagged: {
            "found": True,
            "txn_id": flagged,
            "card_id": card_id,
            "customer_id": customer_id,
            "ts": "2016-11-12 00:46:24",
            "amt": 259.98,
            "product_cd": "W",
            "channel": "online",
            "risk_score": 0.57,
            "addr1": "300",
            "device_key": "dev1",
            "device_type": "mobile",
            "device_info": "SAMSUNG SM-G892A Build/NRD90M",
            "device_new": "New",
            "proxy": "transparent",
            "c_features": {}, "d_features": {}, "m_features": {},
            "v_features_nonnull_count": 3,
        }
    }
    card_window = [
        {"txn_id": "T3450625", "ts": "2016-11-12 00:10:00", "amt": 1.10, "product_cd": "W", "channel": "online", "risk_score": 0.3, "addr1": "300", "device_key": "dev1"},
        {"txn_id": "T3450626", "ts": "2016-11-12 00:20:00", "amt": 2.40, "product_cd": "W", "channel": "online", "risk_score": 0.3, "addr1": "300", "device_key": "dev1"},
        {"txn_id": "T3450627", "ts": "2016-11-12 00:35:00", "amt": 0.95, "product_cd": "W", "channel": "online", "risk_score": 0.3, "addr1": "300", "device_key": "dev1"},
        {"txn_id": flagged, "ts": "2016-11-12 00:46:24", "amt": 259.98, "product_cd": "W", "channel": "online", "risk_score": 0.57, "addr1": "300", "device_key": "dev1"},
    ]
    profiles = {
        customer_id: {
            "customer_id": customer_id, "found": True, "cards": [card_id],
            "txn_count": 4, "total_amt": 264.43, "channels_used": ["online"],
            "prior_closed_cases": [],
        }
    }
    device_neighbors = {
        "dev1": {"device_key": "dev1", "device_info": "SAMSUNG SM-G892A Build/NRD90M",
                  "cards": [card_id, "C08877-K1"], "customers": [customer_id, "C08877"], "txn_count": 6}
    }
    similar_cases = [
        {"case_id": "CC-0141", "case_source": "historical", "score": 0.81, "pattern": "card_testing", "outcome": "confirmed_fraud", "exposure_usd": 200.0, "analyst_notes": "similar card testing"}
    ]
    return FakeGraphClient(txns, {card_id: card_window}, profiles, device_neighbors, similar_cases=similar_cases), flagged, card_id, customer_id


def test_card_testing_end_to_end_denied():
    client, flagged, card_id, customer_id = _card_testing_fixture()
    llm = FakeLLM(
        synthesis_response={
            "pattern": "card_testing",
            "pattern_description": "",
            "verdict": "uncertain",
            "fraud_probability_adjustment": 0.05,
            "summary": "Card testing sequence detected on this card.",
            "additional_evidence_claims": [],
            "first_suspicious_txn_id": "T3450625",
        },
        sar_response={"narrative": "On 2016-11-12, card C04570-K1 was used for a card-testing sequence...", "subjects": [customer_id, card_id]},
    )
    case_row = {
        "case_id": "HHG-017",
        "trigger_type": "risk_score",
        "trigger_text": "Real-time model scored transaction 3450629 at 0.57.",
        "flagged_txn_id": "3450629",
        "card_id": card_id,
        "customer_id": customer_id,
        "risk_score": 0.57,
    }
    answer = run_investigation(case_row, client, llm)

    assert answer.case_id == "HHG-017"
    assert answer.case.pattern == "card_testing"
    assert flagged in answer.case.affected_txn_ids
    assert answer.case.exposure_usd > 0
    # card testing detector should have fired and, combined with a denied
    # simulated response (since a real pattern matched), pushed toward BLOCK/CREATE_CASE
    action_names = {a.action for a in answer.next_best_actions.final}
    assert ActionName.CREATE_CASE in action_names or ActionName.DECLINE_TRANSACTION in action_names
    assert len(answer.evidence_requests) >= 0
    assert answer.tool_calls > 0
    assert client.upserted, "case should have been written to the graph"
    print(answer.model_dump_json(indent=2)[:2000])
