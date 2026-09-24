"""
Unit tests for graph/tigergraph_reshape.py — the one part of the (live,
untestable-without-credentials) TigerGraph backends that IS fully testable
offline: pure data transformation from pyTigerGraph's documented PRINT
result shape into the same dicts graph/local_backend.py returns. Fixtures
below hand-build that raw shape exactly as pyTigerGraph/REST++ documents it
(vertex sets -> [{"v_id":..., "v_type":..., "attributes": {...}}, ...],
scalars/accumulators -> the value directly), matching each query's PRINT
statement in graph/gsql/queries.gsql field-for-field.
"""
from __future__ import annotations

from graph import tigergraph_reshape as reshape


def _v(v_id, v_type, **attrs):
    return {"v_id": v_id, "v_type": v_type, "attributes": attrs}


def test_reshape_transaction_details_found():
    raw = [
        {
            "Txn": [
                _v(
                    "T3013960",
                    "Transaction",
                    ts="2018-01-05 12:00:00",
                    amt=250.0,
                    product_cd="W",
                    channel="online",
                    risk_score=0.8,
                    addr1="123",
                    addr2="87",
                    dist1=None,
                    dist2=None,
                    p_email_domain="gmail.com",
                    r_email_domain=None,
                    c_features='{"C1": 1}',
                    d_features="{}",
                    m_features="{}",
                    v_features='{"V1": 1, "V2": 2}',
                )
            ]
        },
        {"Card_": [_v("C08623-K1", "Card", customer_id="C08623")]},
        {"Cust": [_v("C08623", "Customer")]},
        {"Dev": [_v("a8ae52df946b", "DeviceProfile", device_info="Windows", device_type="desktop", os="Windows 10")]},
        {"Region": [_v("123", "BillingRegion", country_code="87")]},
        {"PEmail": [_v("gmail.com", "EmailDomain")]},
    ]
    out = reshape.reshape_transaction_details(raw, "T3013960")
    assert out["found"] is True
    assert out["card_id"] == "C08623-K1"
    assert out["customer_id"] == "C08623"
    assert out["amt"] == 250.0
    assert out["device_key"] == "a8ae52df946b"
    assert out["device_info"] == "Windows"
    assert out["p_email_domain"] == "gmail.com"
    assert out["c_features"] == {"C1": 1}
    assert out["v_features_nonnull_count"] == 2


def test_reshape_transaction_details_not_found():
    out = reshape.reshape_transaction_details([{"Txn": []}], "T999")
    assert out == {"found": False, "txn_id": "T999"}


def test_reshape_card_window():
    raw = [
        {"AllTxns": [_v("T1", "Transaction", ts="t1", amt=1.0, channel="online")]},
        {
            "Windowed": [
                _v("T1", "Transaction", ts="t1", amt=1.0, channel="online"),
                _v("T2", "Transaction", ts="t2", amt=200.0, channel="online"),
            ]
        },
    ]
    out = reshape.reshape_card_window(raw, "C1-K1")
    assert out["card_id"] == "C1-K1"
    assert out["count"] == 2
    assert [t["txn_id"] for t in out["transactions"]] == ["T1", "T2"]
    assert out["transactions"][1]["amt"] == 200.0


def test_reshape_customer_profile_with_channels_and_total():
    raw = [
        {"Cust": [_v("C1", "Customer")]},
        {"Cards": [_v("C1-K1", "Card"), _v("C1-K2", "Card")]},
        {"txn_count": 5},
        {"Cases": [_v("CC-1", "Case")]},
        {"channels_used": ["online", "in_person"]},
        {"total_amt": 543.21},
    ]
    out = reshape.reshape_customer_profile(raw, "C1")
    assert out["found"] is True
    assert out["cards"] == ["C1-K1", "C1-K2"]
    assert out["txn_count"] == 5
    assert out["channels_used"] == ["in_person", "online"]
    assert out["total_amt"] == 543.21
    assert out["prior_closed_cases"] == ["CC-1"]


def test_reshape_customer_profile_not_found():
    out = reshape.reshape_customer_profile([{"Cust": []}], "C999")
    assert out == {"customer_id": "C999", "found": False}


def test_reshape_device_neighbors_over_cap_still_returns_all_cards():
    """The MAX_MEANINGFUL_SHARED_CARDS cap lives in agent/investigation.py,
    not the reshape layer — reshape must return everything the graph gave
    it; investigation.py decides what's meaningful."""
    raw = [
        {"Dev": [_v("dev1", "DeviceProfile", device_info="Windows")]},
        {"Cards": [_v(f"C{i}-K1", "Card") for i in range(20)]},
        {"Customers": [_v(f"C{i}", "Customer") for i in range(20)]},
        {"txn_count": 500},
    ]
    out = reshape.reshape_device_neighbors(raw, "dev1")
    assert len(out["cards"]) == 20
    assert out["txn_count"] == 500


def test_reshape_card_region_history():
    raw = [
        {
            "Regions": [
                {"v_id": "123", "v_type": "BillingRegion", "attributes": {"region_code": "123", "@txn_count": 7}},
                {"v_id": "456", "v_type": "BillingRegion", "attributes": {"region_code": "456", "@txn_count": 1}},
            ]
        }
    ]
    out = reshape.reshape_card_region_history(raw, "C1-K1")
    assert out["regions"] == {"123": 7, "456": 1}


def test_reshape_velocity():
    raw = [
        {"Recent": [_v("T1", "Transaction", ts="t1", amt=2.0), _v("T2", "Transaction", ts="t2", amt=3.0)]},
        {"total_amt": 5.0},
    ]
    out = reshape.reshape_velocity(raw, "C1-K1")
    assert out["count"] == 2
    assert out["total_amt"] == 5.0


def test_reshape_similar_cases():
    raw = [{"@@top": [{"case_id": "CC-2204", "score": 0.91}, {"case_id": "CC-2151", "score": 0.83}]}]
    out = reshape.reshape_similar_cases(raw)
    assert out == [{"case_id": "CC-2204", "score": 0.91}, {"case_id": "CC-2151", "score": 0.83}]


def test_reshape_similar_cases_empty():
    assert reshape.reshape_similar_cases([{"@@top": []}]) == []
