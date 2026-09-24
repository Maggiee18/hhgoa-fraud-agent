"""
Thin data-access layer between the Streamlit dashboard (frontend/app.py) and
the agent pipeline. Deliberately NOT a network API — for a single-process
demo app, importing agent/graph directly is simpler and has fewer moving
parts than standing up a separate backend service (per the "keep
architecture simple" engineering rule). This module is what makes that
seam explicit and swappable later if a real API is ever wanted.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

CASES_DIR_DEFAULT = os.getenv("CASES_OUTPUT_DIR", "./cases")
DATA_DIR = os.getenv("DATA_DIR", "./data")


def list_case_files(cases_dir: str = CASES_DIR_DEFAULT) -> List[str]:
    """case_ids that already have a written cases/<case_id>.json answer."""
    if not os.path.isdir(cases_dir):
        return []
    paths = sorted(glob.glob(os.path.join(cases_dir, "HHG-*.json")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def load_case_answer(case_id: str, cases_dir: str = CASES_DIR_DEFAULT) -> Optional[Dict[str, Any]]:
    path = os.path.join(cases_dir, f"{case_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_case_pack(case_pack_path: Optional[str] = None) -> pd.DataFrame:
    path = case_pack_path or os.path.join(DATA_DIR, "case_pack.csv")
    return pd.read_csv(path)


def run_new_case(case_id: str, backend: Optional[str], case_pack_path: Optional[str] = None):
    """Runs one case_pack row through the real pipeline (real LLM — requires
    ANTHROPIC_API_KEY). Raises on missing key/creds rather than silently
    falling back, so the dashboard can show the user exactly what's
    missing instead of a confusing downstream error."""
    from agent.graph_agent import run_investigation
    from agent.llm import LLM
    from graph.client import get_graph_client

    case_pack = load_case_pack(case_pack_path)
    row = case_pack[case_pack["case_id"] == case_id]
    if row.empty:
        raise ValueError(f"case_id {case_id} not found in case_pack.csv")

    client = get_graph_client(force_backend=backend)
    llm = LLM()
    return run_investigation(row.iloc[0].to_dict(), client, llm)


def save_case_answer(answer, cases_dir: str = CASES_DIR_DEFAULT) -> str:
    os.makedirs(cases_dir, exist_ok=True)
    path = os.path.join(cases_dir, f"{answer.case_id}.json")
    with open(path, "w") as f:
        f.write(answer.model_dump_json(indent=2))
    return path
