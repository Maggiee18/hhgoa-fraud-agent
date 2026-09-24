#!/usr/bin/env python3
"""
Runs all 20 cases from case_pack.csv and writes cases/<case_id>.json for
each, per the README's submission format: "Submit the 20 answer files ...
in a folder called cases/ in your repository."

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --backend local --limit 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agent.graph_agent import run_investigation
from agent.llm import LLM
from graph.client import get_graph_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--case-pack", default=os.path.join(os.getenv("DATA_DIR", "./data"), "case_pack.csv"))
    ap.add_argument("--out-dir", default=os.getenv("CASES_OUTPUT_DIR", "./cases"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated case_ids to run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    case_pack = pd.read_csv(args.case_pack)
    if args.only:
        wanted = set(args.only.split(","))
        case_pack = case_pack[case_pack["case_id"].isin(wanted)]
    if args.limit:
        case_pack = case_pack.head(args.limit)

    client = get_graph_client(force_backend=args.backend)
    llm = LLM()

    results = []
    t0 = time.time()
    for _, row in case_pack.iterrows():
        case_row = row.to_dict()
        case_id = case_row["case_id"]
        print(f"=== {case_id} ===", flush=True)
        try:
            answer = run_investigation(case_row, client, llm)
            out_path = os.path.join(args.out_dir, f"{case_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(answer.model_dump_json(indent=2))
            print(
                f"  verdict={answer.case.verdict} pattern={answer.case.pattern} "
                f"prob={answer.case.fraud_probability:.2f} actions={[a.action for a in answer.next_best_actions.final]} "
                f"tool_calls={answer.tool_calls} tokens={answer.tokens} latency={answer.latency_s:.1f}s"
            )
            results.append({"case_id": case_id, "status": "ok"})
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"case_id": case_id, "status": "error", "error": str(e)})

    dt = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} cases completed in {dt:.1f}s")
    summary_path = os.path.join(args.out_dir, "_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_s": dt}, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
