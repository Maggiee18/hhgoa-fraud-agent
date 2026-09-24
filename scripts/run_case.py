#!/usr/bin/env python3
"""Run one case from case_pack.csv end to end and print/save its answer JSON.

Usage:
    python scripts/run_case.py HHG-017
    python scripts/run_case.py HHG-017 --backend local
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agent.graph_agent import run_investigation
from agent.llm import LLM
from graph.client import get_graph_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--backend", default=None, help="override GRAPH_BACKEND (tigergraph|local)")
    ap.add_argument("--case-pack", default=os.path.join(os.getenv("DATA_DIR", "./data"), "case_pack.csv"))
    ap.add_argument("--out", default=None, help="write answer JSON here (defaults to stdout only)")
    args = ap.parse_args()

    case_pack = pd.read_csv(args.case_pack)
    row = case_pack[case_pack["case_id"] == args.case_id]
    if row.empty:
        print(f"case_id {args.case_id} not found in {args.case_pack}", file=sys.stderr)
        sys.exit(1)
    case_row = row.iloc[0].to_dict()

    client = get_graph_client(force_backend=args.backend)
    llm = LLM()

    answer = run_investigation(case_row, client, llm)
    out_json = answer.model_dump_json(indent=2)
    print(out_json)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
