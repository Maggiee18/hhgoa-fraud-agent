"""
Analyst dashboard for the HHGOA fraud investigation agent. Run with:

    streamlit run frontend/app.py

Reads finished cases/<case_id>.json answer files (the actual submission
artifacts) and can also trigger a brand-new investigation against
case_pack.csv live, using whichever GRAPH_BACKEND / ANTHROPIC_API_KEY are
configured in the environment. Deliberately a single Streamlit process that
imports agent/graph directly (see backend/case_store.py) rather than a
separate API service — simplest thing that works for a demo/analyst tool.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.case_store import (
    CASES_DIR_DEFAULT,
    DATA_DIR,
    list_case_files,
    load_case_answer,
    load_case_pack,
    run_new_case,
    save_case_answer,
)

st.set_page_config(page_title="HHGOA Fraud Investigation Agent", layout="wide")

VERDICT_COLOR = {"fraud": "#d62728", "legitimate": "#2ca02c", "uncertain": "#ff7f0e"}
STATUS_LABEL = {
    "open": "Open",
    "closed_fraud": "Closed — Fraud",
    "closed_legitimate": "Closed — Legitimate",
    "escalated": "Escalated",
}


# --- Sidebar: case selection / live run -------------------------------------

st.sidebar.title("HHGOA Fraud Agent")
cases_dir = st.sidebar.text_input("Answers directory", value=CASES_DIR_DEFAULT)
available = list_case_files(cases_dir)

st.sidebar.markdown("---")
st.sidebar.subheader("Run a new investigation")
try:
    case_pack = load_case_pack()
    candidate_ids = case_pack["case_id"].tolist()
except Exception as e:
    case_pack = None
    candidate_ids = []
    st.sidebar.warning(f"Could not load case_pack.csv: {e}")

new_case_id = st.sidebar.selectbox("case_pack.csv trigger", options=["(none)"] + candidate_ids)
backend_choice = st.sidebar.selectbox("Graph backend", options=["(env default)", "local", "tigergraph"])
if st.sidebar.button("Investigate", disabled=(new_case_id == "(none)")):
    with st.spinner(f"Investigating {new_case_id}..."):
        try:
            backend = None if backend_choice == "(env default)" else backend_choice
            answer = run_new_case(new_case_id, backend)
            path = save_case_answer(answer, cases_dir)
            st.sidebar.success(f"Wrote {path}")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Investigation failed: {e}")
            if "ANTHROPIC_API_KEY" in str(e):
                st.sidebar.info("Set ANTHROPIC_API_KEY in .env to enable live investigations.")

st.sidebar.markdown("---")
if not available:
    st.sidebar.info("No case answers yet. Run `python scripts/run_benchmark.py` or use 'Investigate' above.")
    selected_case_id = None
else:
    selected_case_id = st.sidebar.radio("Investigated cases", options=available)


# --- Main -------------------------------------------------------------------

if not selected_case_id:
    st.title("HHGOA Fraud Investigation Agent")
    st.write(
        "No investigated cases found yet. Use the sidebar to run one of the "
        "20 case_pack.csv triggers, or point the answers directory at an "
        "existing `cases/` folder."
    )
    st.stop()

data = load_case_answer(selected_case_id, cases_dir)
if data is None:
    st.error(f"Could not load {selected_case_id}.json from {cases_dir}")
    st.stop()

case = data["case"]
verdict = case["verdict"]
status = case["status"]
prob = case["fraud_probability"]

st.title(f"{selected_case_id}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", STATUS_LABEL.get(status, status))
col2.metric("Verdict", verdict.capitalize())
col3.metric("Pattern", case["pattern"].replace("_", " "))
col4.metric("Exposure (USD)", f"${case['exposure_usd']:,.2f}")

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Fraud probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": VERDICT_COLOR.get(verdict, "#1f77b4")},
            "steps": [
                {"range": [0, 20], "color": "#e8f5e9"},
                {"range": [20, 60], "color": "#fff3e0"},
                {"range": [60, 100], "color": "#ffebee"},
            ],
        },
    )
)
fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))

top_l, top_r = st.columns([1, 2])
with top_l:
    st.plotly_chart(fig, use_container_width=True)
with top_r:
    st.markdown("**Summary**")
    st.write(case["summary"])
    if case.get("pattern_description"):
        st.caption(case["pattern_description"])

tabs = st.tabs(
    ["Evidence", "Uncertainty & Timeline", "Next Best Action", "Fraud Graph", "Case Memory", "SAR"]
)

# --- Evidence -----------------------------------------------------------
with tabs[0]:
    st.subheader("Evidence")
    ev = case.get("evidence", [])
    if ev:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "claim": e["claim"],
                        "source": e["source"],
                        "ref": e["ref"],
                        "entity_ids": ", ".join(e.get("entity_ids", [])),
                    }
                    for e in ev
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No evidence recorded.")
    st.markdown("**Affected transactions**")
    st.code(", ".join(case.get("affected_txn_ids", [])) or "(none)")
    st.markdown(f"**First suspicious transaction:** `{case.get('first_suspicious_txn_id') or '(none)'}`")

# --- Uncertainty & Timeline -----------------------------------------------
with tabs[1]:
    st.subheader("Evidence-gathering timeline")
    reqs = data.get("evidence_requests", [])
    if reqs:
        st.dataframe(
            pd.DataFrame(
                [
                    {"after_step": r["asked_after_step"], "type": r["type"], "assumed_response": r["assumed_response"]}
                    for r in reqs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No additional evidence was requested — enough was available after the initial gather.")
    st.markdown("**Why the investigation stopped**")
    st.info(data.get("stop_reason", "(not recorded)"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Tool calls", data.get("tool_calls", 0))
    m2.metric("Tokens", data.get("tokens", 0))
    m3.metric("Latency (s)", f"{data.get('latency_s', 0):.1f}")

# --- Next Best Action -------------------------------------------------------
with tabs[2]:
    st.subheader("Next best action")
    nba = data["next_best_actions"]

    def _actions_df(actions):
        if not actions:
            return None
        return pd.DataFrame(
            [{"action": a["action"], "approval_route": a["route"], "reason": a["reason"]} for a in actions]
        )

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**Before additional evidence**")
        df_i = _actions_df(nba.get("initial"))
        if df_i is not None:
            st.dataframe(df_i, use_container_width=True, hide_index=True)
        else:
            st.write("(none)")
    with ac2:
        st.markdown("**After additional evidence**")
        df_f = _actions_df(nba.get("final"))
        if df_f is not None:
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        else:
            st.write("(none)")
    st.markdown("**What changed**")
    st.write(nba.get("what_changed", "nothing"))
    st.caption(
        "Approval routes: `auto` = agent may execute directly, `L1`/`L2` = requires human "
        "approval at that tier, per the bank's fraud policy."
    )

# --- Fraud Graph -------------------------------------------------------
with tabs[3]:
    st.subheader("Fraud graph")
    conn_cards = case.get("connected_card_ids", [])
    conn_devices = case.get("connected_device_profiles", [])
    if not conn_cards and not conn_devices:
        st.write("No connected cards or devices met the shared-origin threshold for this case.")
    else:
        try:
            from pyvis.network import Network

            net = Network(height="420px", width="100%", bgcolor="#ffffff", font_color="#222")
            center = case_pack[case_pack["case_id"] == selected_case_id]["card_id"].iloc[0] if case_pack is not None and selected_case_id in case_pack["case_id"].values else "flagged card"
            net.add_node(center, label=str(center), color=VERDICT_COLOR.get(verdict, "#1f77b4"), size=30)
            for c in conn_cards:
                net.add_node(c, label=c, color="#9e9e9e", size=18)
                net.add_edge(center, c, label="shared origin")
            for d in conn_devices:
                net.add_node(d, label=d, color="#607d8b", shape="box", size=18)
                net.add_edge(center, d, label="from device")
            html_path = f"/tmp/graph_{selected_case_id}.html"
            net.write_html(html_path, open_browser=False, notebook=False)
            with open(html_path) as f:
                st.components.v1.html(f.read(), height=440, scrolling=True)
        except Exception as e:
            st.warning(f"Graph render unavailable ({e}); connected entities: cards={conn_cards}, devices={conn_devices}")

# --- Case Memory -------------------------------------------------------
with tabs[4]:
    st.subheader("Similar prior cases (case memory)")
    similar = case.get("similar_prior_cases", [])
    if not similar:
        st.write("No similar prior cases were retrieved.")
    else:
        try:
            closed = pd.read_csv(os.path.join(DATA_DIR, "closed_cases_history.csv"))
            rows = closed[closed["case_id"].isin(similar)][
                ["case_id", "pattern", "outcome", "exposure_usd", "n_txns", "opened_at", "closed_at"]
            ]
            # preserve retrieval order / rank
            rows = rows.set_index("case_id").reindex(similar).reset_index()
            st.dataframe(rows, use_container_width=True, hide_index=True)
        except Exception as e:
            st.write(similar)
            st.caption(f"(could not join closed_cases_history.csv: {e})")

# --- SAR -----------------------------------------------------------------
with tabs[5]:
    st.subheader("Suspicious Activity Report")
    sar = data["sar"]
    if not sar.get("file"):
        st.write("Not filed.")
        st.caption(f"Reason: {sar.get('reason', '(not recorded)')}")
    else:
        st.success(f"Filed — {sar.get('reason', '')}")
        st.markdown("**Narrative**")
        st.write(sar.get("narrative", ""))
        st.markdown(f"**Subjects:** {', '.join(sar.get('subjects', [])) or '(none)'}")
        st.markdown(f"**Total amount:** ${sar.get('total_amount_usd', 0):,.2f}")
        st.markdown(f"**Activity dates:** {', '.join(sar.get('activity_dates', [])) or '(none)'}")
