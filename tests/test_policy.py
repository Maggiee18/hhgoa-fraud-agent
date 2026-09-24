from agent.policy import PolicyFacts, decide, route_for, stop_investigation
from agent.schemas import ActionName, ApprovalRoute


def test_r1_weak_signal_verify_before_block():
    facts = PolicyFacts(
        fraud_probability=0.61,
        pattern="none",
        exposure_usd=77.07,
        single_weak_signal=True,
    )
    d = decide(facts)
    names = [a.action for a in d.actions]
    assert ActionName.VERIFY_WITH_CUSTOMER in names
    assert ActionName.BLOCK_CARD not in names
    assert "R1" in d.rules_applied


def test_r2_customer_denies_low_exposure_L1():
    facts = PolicyFacts(
        fraud_probability=0.86,
        pattern="card_testing",
        exposure_usd=268.43,
        single_weak_signal=False,
        customer_response="denied",
        shared_origin="device profile D000731",
        connected_card_ids=["C00877-K1"],
    )
    d = decide(facts)
    block = next(a for a in d.actions if a.action == ActionName.BLOCK_CARD)
    assert block.route == ApprovalRoute.L1  # <= 2500
    assert any(a.action == ActionName.FILE_REPORT for a in d.actions)  # shared origin
    assert any(a.action == ActionName.MONITOR_CONNECTED_CARDS for a in d.actions)


def test_r2_high_exposure_routes_L2():
    r = route_for(ActionName.BLOCK_CARD, exposure_usd=5000)
    assert r == ApprovalRoute.L2


def test_r3_customer_confirms_closes():
    facts = PolicyFacts(
        fraud_probability=0.3,
        pattern="none",
        exposure_usd=100,
        single_weak_signal=True,
        customer_response="confirmed",
    )
    d = decide(facts)
    assert [a.action for a in d.actions] == [ActionName.CLOSE_NO_FRAUD]
    assert d.rules_applied == ["R3"]


def test_r5_card_testing_with_cleared_purchase_blocks():
    facts = PolicyFacts(
        fraud_probability=0.72,
        pattern="card_testing",
        exposure_usd=268.43,
        single_weak_signal=False,
        card_testing_sequence=True,
        purchase_over_100_cleared=True,
    )
    d = decide(facts)
    names = [a.action for a in d.actions]
    assert ActionName.DECLINE_TRANSACTION in names
    assert ActionName.STEP_UP_AUTH in names
    assert ActionName.BLOCK_CARD in names


def test_r7_disputed_recurring_never_blocks():
    facts = PolicyFacts(
        fraud_probability=0.4,
        pattern="none",
        exposure_usd=50,
        single_weak_signal=False,
        disputed_but_recurring=True,
    )
    d = decide(facts)
    names = [a.action for a in d.actions]
    assert ActionName.BLOCK_CARD not in names
    assert ActionName.DECLINE_TRANSACTION not in names
    assert ActionName.VERIFY_WITH_CUSTOMER in names


def test_r8_escalate_uncertain_and_exposed():
    facts = PolicyFacts(
        fraud_probability=0.5,
        pattern="none",
        exposure_usd=800,
        single_weak_signal=False,
        verdict_uncertain=True,
    )
    d = decide(facts)
    assert any(a.action == ActionName.ESCALATE_TO_ANALYST for a in d.actions)


def test_route_for_all_auto_actions():
    from agent.policy import AUTO_ACTIONS
    for a in AUTO_ACTIONS:
        assert route_for(a) == ApprovalRoute.auto


def test_stop_investigation_decisive_probability():
    assert stop_investigation(0.9, independent_evidence_count=2, verification_settled=False, further_steps_unlikely_to_change=False)
    assert stop_investigation(0.9, independent_evidence_count=1, verification_settled=False, further_steps_unlikely_to_change=False) is None
    assert stop_investigation(0.5, independent_evidence_count=0, verification_settled=True, further_steps_unlikely_to_change=False)
    assert stop_investigation(0.5, independent_evidence_count=0, verification_settled=False, further_steps_unlikely_to_change=False) is None
