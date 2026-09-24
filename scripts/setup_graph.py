#!/usr/bin/env python3
"""
Creates the HHGOA_Fraud graph schema and installs the query set on a real
TigerGraph Savanna/CE instance, then loads the four CSVs.

Requires TG_HOST and TG_SECRET in .env — see .env.example. This is the one
step that genuinely needs to run somewhere with network access to
*.tgcloud.io; it could not be executed or tested from the development
sandbox (see docs/DECISIONS.md) so run it yourself and report any GSQL
syntax issues — they're plausible on a first real run against a specific
TigerGraph version.

Savanna auth note: unlike classic TigerGraph Cloud, Savanna does not accept
username/password — only a Database Secret (tgCloud=True + gsqlSecret in
the constructor; no username/password passed at all). Passing
username/password (even the old "tigergraph" default) triggers a
'User authentication failed' error before the secret is ever used.

Savanna auto-suspend note: an idle workspace suspends itself, and the
first request after that wakes it but can surface as a confusing
Bad-Gateway/HTML-in-JSON error rather than a clean timeout. run_gsql_file
retries for a couple of minutes before giving up, so a cold-start wake-up
doesn't look like a real failure.

Catalog-reset note: GSQL's CREATE VERTEX/EDGE/GRAPH statements are not
idempotent. If an earlier run of --schema failed partway through (a syntax
error a few lines in, say), the vertex/edge types it already created stay
in the workspace's catalog, so rerunning the same schema.gsql collides
with itself ("Card is used by another object!"). --reset runs `DROP ALL`
first to wipe the catalog clean before recreating everything. Safe to use
here because nothing has been loaded into a real graph yet; the only
casualty is the pre-built sample solution kit a new Savanna workspace
ships with (e.g. Transaction_Fraud), which this project doesn't use.

Usage:
    python scripts/setup_graph.py --reset --all      # first successful run, or after any --schema failure
    python scripts/setup_graph.py --schema --queries --load
    python scripts/setup_graph.py --load-only   # schema/queries already installed
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def connect():
    """Savanna-specific: no username/password (it rejects them outright —
    'User authentication failed' — even the legacy "tigergraph" default).
    Auth is secret-only, passed straight into the constructor with
    tgCloud=True, matching how other teams on this same challenge connect
    to Savanna. graphname starts empty since the graph doesn't exist yet
    on a first --schema run."""
    import pyTigerGraph as tg

    host = os.environ["TG_HOST"]
    secret = os.environ.get("TG_SECRET", "")
    if not secret:
        raise RuntimeError("TG_SECRET is not set in .env — Savanna auth needs a Database Secret.")
    conn = tg.TigerGraphConnection(host=host, graphname="", gsqlSecret=secret, tgCloud=True)
    conn.apiToken = conn.getToken(secret)[0]
    return conn


def _with_wakeup_retry(fn, attempts: int = 6, delay_s: float = 20.0):
    """Savanna suspends an idle workspace; the first request after that
    wakes it but can come back as a Bad-Gateway/HTML-in-JSON error instead
    of a clean timeout. Retry for ~2 minutes before treating it as real."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == attempts - 1:
                raise
            print(f"  (attempt {attempt + 1}/{attempts} failed: {e} — retrying in {delay_s:.0f}s, "
                  f"workspace may be waking from auto-suspend)")
            time.sleep(delay_s)
    raise last_exc  # pragma: no cover


def reset_catalog(conn):
    """Wipes the whole workspace catalog (DROP ALL). See the module
    docstring's Catalog-reset note for why this is needed and why it's safe
    to run here."""
    print("--- resetting catalog (DROP ALL) ---")
    result = _with_wakeup_retry(lambda: conn.gsql("DROP ALL"))
    print(result)


def run_gsql_file(conn, path: str):
    with open(path) as f:
        script = f.read()
    print(f"--- running {path} ---")
    result = _with_wakeup_retry(lambda: conn.gsql(script))
    print(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--all", action="store_true", help="schema + queries + load")
    ap.add_argument("--reset", action="store_true", help="DROP ALL before schema (see module docstring)")
    args = ap.parse_args()
    if args.all:
        args.schema = args.queries = args.load = True

    conn = connect()
    conn.graphname = os.environ.get("TG_GRAPH_NAME", "HHGOA_Fraud")

    if args.reset:
        reset_catalog(conn)
        # DROP ALL invalidates the token we already fetched; get a fresh one.
        secret = os.environ.get("TG_SECRET", "")
        conn.apiToken = conn.getToken(secret)[0] if secret else conn.apiToken

    if args.schema:
        run_gsql_file(conn, os.path.join(ROOT, "graph", "schema", "schema.gsql"))
    if args.queries:
        run_gsql_file(conn, os.path.join(ROOT, "graph", "gsql", "queries.gsql"))
    if args.load:
        from graph.load_data import load_all

        # Re-fetch the token now that conn.graphname points at the graph
        # that just got created (tokens are graph-scoped).
        secret = os.environ.get("TG_SECRET", "")
        conn.apiToken = conn.getToken(secret)[0] if secret else conn.apiToken
        _with_wakeup_retry(lambda: load_all(conn))

    print("Done.")


if __name__ == "__main__":
    main()
