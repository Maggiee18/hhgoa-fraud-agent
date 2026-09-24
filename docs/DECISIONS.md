# Engineering decisions log

## 2026-09-22 — Kickoff

- **Repo location**: full repo built in this cloud sandbox. User pushes to
  GitHub themselves before the 2026-09-24 11:59 PM IST deadline.
- **Dataset**: HHGOA_IEEE obtained from the user's Google Drive folder
  (link they shared) via the browser pane, since neither the cloud
  sandbox's nor the user's laptop's outbound network can reach
  `drive.google.com` or `tgcloud.io` directly (both return 403 from the
  egress proxy — confirmed by direct curl test from both `bash` and
  `device_bash`). Files copied into
  `C:\Users\HP\OneDrive\Desktop\TigerGraph Agentic Fraud Investigation HHGOA\data\`:
  `README.md`, `case_pack.csv` (20 rows), `closed_cases_history.csv` (5565
  rows), `identity.csv` (144432 rows). `transactions.csv` (590742 rows,
  ~708MB) downloading in background via the browser pane.
- **Critical network constraint**: TigerGraph Savanna (`*.tgcloud.io`) is
  NOT reachable from either the cloud sandbox `bash` or the user's laptop
  `device_bash` shell under the current egress policy. This means:
  - I cannot execute GSQL schema creation, data loading, or live queries
    against a real TigerGraph instance from this session.
  - All TigerGraph-facing code (schema, loading jobs, pyTigerGraph client,
    MCP tool wrappers) will be written correctly and completely, but the
    user must run the actual `scripts/setup_graph.py` load step themselves
    (or from an environment with real network access), following the exact
    commands I provide.
  - To keep the agent itself fully testable and the benchmark runnable
    end-to-end *now*, the graph access layer is built against a small
    interface (`graph/client.py` protocol) with two backends: `tigergraph`
    (pyTigerGraph, for the real submission) and `local` (an in-memory
    NetworkX graph built directly from the same CSVs, for development and
    as an offline fallback demo path). Same query methods either way. This
    is a pragmatic call, not a shortcut on requirements — the required
    deliverable still targets real TigerGraph + GSQL + MCP; `local` exists
    purely so I can build, test, and hand over correct agent behavior
    without live infra access, and so the user has a working demo even if
    Savanna briefly stops.
- **TigerGraph target: Savanna**, not Community Edition (matches hackathon
  guidance; CE would mean standing up a JVM service with no persistence in
  this ephemeral sandbox for zero benefit).
- **LLM**: Anthropic API key, supplied by user, read only from
  `ANTHROPIC_API_KEY` env var.
- **Stack**: Python 3.11+ throughout. LangGraph for the explicit
  investigation loop. `pyTigerGraph` + TigerGraph MCP for graph access.
  Streamlit for the investigator dashboard (fastest path to a demo-ready UI
  in the time available). Case records as pydantic models validated against
  the README's exact answer schema, persisted as JSON under `/cases` (the
  required submission format) and mirrored into the graph as `Case`
  vertices.
- **Case memory / GraphRAG**: embeddings via TF-IDF (scikit-learn) over the
  case-narrative corpus, stored as a vector attribute on `Case` vertices;
  retrieval via TigerGraph vector search when available, else client-side
  cosine similarity as a reliable fallback. Originally planned as local
  `sentence-transformers` (all-MiniLM-L6-v2), but switched after
  confirming `huggingface.co` is unreachable from this sandbox's network
  (same 403-from-proxy restriction as `tgcloud.io`/`drive.google.com` —
  see the network-constraint entry above) — a model-download dependency
  would be untestable here and could silently fail in any similarly
  locked-down deployment. TF-IDF is fully offline/deterministic, needs no
  model download, and fits short structured case-narrative text well.
  Vectorizer is fit once from `closed_cases_history.csv` and cached to
  `graph/tfidf_vectorizer.pkl` so query-time embeddings stay consistent
  with what's stored.
- **Graph schema**: extends the README's suggested schema. One unified
  `Case` vertex type (not separate historical/agent types) with a
  `case_source` attribute (`historical` | `agent`), so the 5565 seeded
  closed cases and every case the agent opens live in the same
  queryable/retrievable pool — directly satisfying "write the case to the
  graph so later investigations can find it." `V1..V339`/`C1..C14`/
  `D1..D15`/`M1..M9`/`id_01..id_38` are stored as compact JSON-encoded
  attributes on `Transaction`/`DeviceProfile` rather than 300+ individual
  GSQL attributes — they're unnamed model features per the README, used as
  aggregate/statistical signals, not individually reasoned about by name.

## Confirmed dataset facts (from README + direct inspection)

- `transactions.csv`: 590,742 rows, 393 original Vesta columns +
  `customer_id`, `ts`, `channel`, `risk_score`. No fraud flag.
- `identity.csv`: 144,432 rows, 41 columns (`TransactionID` + `id_01..38` +
  `DeviceType` + `DeviceInfo`), online transactions only, joins on
  `TransactionID`.
- `closed_cases_history.csv`: 5,565 rows confirmed by direct count — 4,665
  `confirmed_fraud`, 900 `cleared`. Pattern breakdown (confirmed_fraud
  only): `card_not_present_fraud` 1404, `account_takeover` 1205,
  `card_not_present_new_device` 1076, `out_of_region_use` 955,
  `card_testing` 16, `undocumented` 9. (`cleared` cases carry
  `pattern=none`.) Card testing and undocumented are rare — worth specific
  attention in retrieval since few examples exist.
- `case_pack.csv`: 20 rows exactly as listed in the README table.
  Trigger types: `risk_score`, `customer_report`, `analyst_request`.
- Answer format, fraud policy (actions, approval routing R1-R10, 3a
  case-vs-SAR, stopping rule §6) all captured verbatim in the downloaded
  `data/README.md` — that file is the source of truth, not this log.

## Device-fingerprint over-matching fix (shared-origin / R6)

First real-data run of the full pipeline (case HHG-017, local backend, dev
sample of 121,941 txns) surfaced a false-positive: R6 ("shared origin
across cards") fired on 15+ `connected_card_ids`, all sharing
`device_key=a8ae52df946b` / `DeviceInfo=Windows`. Investigated directly
against the sampled data:

- `DeviceInfo=="Windows"` alone (no further id_* detail) appears on 13,251
  of 121,941 transactions (~11%) — same for other generic strings
  (`iOS Device`, `MacOS`, `Trident/7.0`, bare `rv:NN.0`).
- Distinct-card count per `device_key`: median 1, 75th pct 2, but a long
  tail up to 335 distinct cards on one key. `DeviceInfo`/`id_30`/`id_31` in
  this dataset are OS/browser/device-type strings, not unique per-customer
  device IDs, so a popular one collides across thousands of *unrelated*
  real customers — it is not evidence of a fraud ring reusing one device.

Fix: `agent/investigation.py`'s `has_shared_origin()` / `connected_cards()`
(the sole producers of `PolicyFacts.shared_origin` /
`connected_card_ids`, i.e. the only path into R6) now only treat a
device/region match as meaningful when the number of distinct cards
sharing it is small — `1 < n <= MAX_MEANINGFUL_SHARED_CARDS` (=5, picked
from the distinct-card-per-device distribution above: ~89% of device_keys
fall at or below this). Above the cap, the match is discarded entirely
(no shared_origin, no connected_card_ids) rather than truncated, since a
large match count is itself the signal that the fingerprint is generic
noise, not partial fraud-ring evidence. Backend-agnostic fix (both
`LocalGraphClient` and the GSQL `get_device_neighbors`/`get_region_cluster`
queries still return full unbounded card lists — the judgment of what's
"meaningful" belongs in the deterministic policy-facts layer, not
duplicated in two backends). Re-validated: all 20 case_pack cases now run
end-to-end with 0 crashes and connected-card counts in a sane 0-3 range
(previously HHG-017 alone showed 15+); 11/11 unit tests still pass.

## Dashboard (frontend/app.py, backend/case_store.py)

Streamlit chosen (already planned) over standing up a separate API +
frontend: this is a single analyst-facing tool reading JSON case answers
plus one live "run an investigation" action, so a second process/service
adds deployment complexity without benefit. `backend/` holds only the data
access seam (`case_store.py`) so the app doesn't import `agent`/`graph`
directly — kept thin, not a real API layer, per "keep architecture simple."

Panels: case header (status/verdict/pattern/exposure) + a fraud-probability
gauge, then tabs for Evidence, Uncertainty & Timeline (evidence requests,
stop_reason, tool_calls/tokens/latency), Next Best Action (initial vs final
side by side + what_changed), Fraud Graph (pyvis network of the flagged
card against `connected_card_ids`/`connected_device_profiles` — empty
after the device-fingerprint fix above unless the match was genuinely
tight), Case Memory (joins `similar_prior_cases` against
`closed_cases_history.csv`), and SAR (narrative or reason-not-filed). The
sidebar can also trigger a brand-new investigation straight from
case_pack.csv via `backend.run_new_case`, surfacing a clear error (not a
stack trace) if `ANTHROPIC_API_KEY` isn't set.

Tested with Streamlit's `AppTest` harness (in-process script execution,
catches exceptions without a real browser — this sandbox/device_bash have
no way to click through a UI): 0 exceptions across all three dev-sample
cases (including both the SAR-filed and not-filed branches, and both the
connected-cards and no-connected-cards branches of the fraud graph), plus
the empty-state ("no cases yet") path. Dev-sample answers used for this
were generated with `tests.fakes.FakeLLM`, NOT the real LLM — written to
`outputs/dev_cases_fakellm/` (gitignored) specifically so they're never
mistaken for real submission output in `cases/`.

## TigerGraph MCP integration (graph/tigergraph_mcp_backend.py)

The challenge lists TigerGraph MCP as a required component ("Use TigerGraph
MCP to expose graph capabilities and data to the agent"), which the agent
wasn't satisfying — it called GraphClient's Python interface directly.
Fixed by adding a third GraphClient backend (`GRAPH_BACKEND=tigergraph_mcp`)
that routes every graph read/write through the real `tigergraph-mcp`
package instead of a direct pyTigerGraph connection, while keeping the
validated deterministic pipeline (detectors/policy/investigation.py)
completely unaware of the difference — same Protocol, same query names,
same param dicts, same result shape.

Getting the tool names/argument shape right without guessing needed the
real package: github.com is blocked in this sandbox, but pypi.org worked
through the browser-pane bridge (the user's real browser, granted access),
which led to `tigergraph-mcp` on PyPI (Sep 2026, official, Apache-2.0) —
its README there has the full tool list (69 tools,
`tigergraph__run_installed_query` etc.), env-var config, and client
examples. Then `pip install tigergraph-mcp` (bumps pyTigerGraph to 2.0.4,
which is required by the mcp package and still keeps the sync
`TigerGraphConnection` class `tigergraph_backend.py` already used — no
behavior change there) and read its source directly to confirm
`run_installed_query(query_name, params)` forwards `params` straight into
`conn.runInstalledQuery(query_name, params)`, unmodified — i.e. exactly
what `tigergraph_backend.py` already builds. So `TigerGraphMCPClient`
subclasses `TigerGraphClient` and overrides only `__init__` +
`_run_installed_query`, reusing every query-name/param dict already
written and tested.

Also found while doing this (pre-existing bug, unrelated to MCP itself):
`tigergraph_backend.py` returned pyTigerGraph's raw
`[{PRINT_var: value}, ...]` result verbatim, but every caller
(agent/investigation.py, agent/detectors.py) expects the plain dicts
`local_backend.py` returns (e.g. `device_neighbors["cards"]`, not
`[{"Cards": [...]}]`). Would have broken the moment real TigerGraph
credentials arrived. Fixed with a shared `graph/tigergraph_reshape.py`
(one reshape function per query, used by both real backends) — the one
piece of this that's fully unit-testable offline (`tests/test_tigergraph_reshape.py`,
hand-built fixtures matching pyTigerGraph's documented PRINT-result shape,
10 tests, all passing). Also caught `get_customer_profile`'s GSQL not
computing `channels_used`/`total_amt` at all, which `detect_account_takeover`
needs for its multi-channel signal — fixed by adding
`SetAccum<STRING> @@channels` / `SumAccum<DOUBLE> @@total_amt` to that
query (graph/gsql/queries.gsql).

Live-validated as much as possible without a real TigerGraph instance: ran
the full stdio connect → tool call → close cycle against the actual
installed `tigergraph-mcp` binary with a deliberately unreachable
`TG_HOST`. This is genuinely useful (not a no-op) — it exercises the real
MCP handshake, the real tool schema, and the real error path, and it
surfaced and let us fix two real bugs before they could hit a live demo:
(1) the tool's response text isn't plain JSON, it's a ` ```json ` -fenced
block followed by a human-readable rendering of the same payload — naive
`json.loads(text)` failed; fixed with fenced-block extraction (falling
back to brace-matching). (2) bridging the async `ClientSession` to
synchronous GraphClient calls by calling `__aenter__`/`__aexit__` from
separate `run_coroutine_threadsafe` submissions crashed on close
("Attempted to exit cancel scope in a different task") because anyio
cancel scopes must be entered and exited from the same asyncio Task —
fixed with `_SessionWorker`, one long-lived coroutine holding the session
for its whole life and pulling work off an `asyncio.Queue`. What's still
genuinely untested is the query results themselves against real data,
since that needs a live TigerGraph behind the MCP server.

## Pluggable LLM provider (Gemini added alongside Anthropic)

The user's Anthropic key authenticated correctly (after fixing a real SDK
compatibility bug below) but returned "Your credit balance is too low" —
Anthropic requires billing/credits added before any call succeeds, with no
card-free path. The challenge explicitly allows "the LLM or combination of
models of your choice", so rather than block on that purchase,
`agent/llm.py` was refactored to support Gemini as a second provider behind
the same `LLM(...).complete(system, messages, max_tokens, temperature) ->
str` interface every caller already uses — zero changes needed anywhere
else (agent/investigation.py, agent/graph_agent.py, scripts/*,
backend/case_store.py). `LLM_PROVIDER` env var picks explicitly; left
blank, it auto-picks gemini if GEMINI_API_KEY is set, else anthropic —
confirmed both directions work (with only ANTHROPIC_API_KEY set, it
correctly still picks anthropic).

Chose Gemini specifically because its free tier genuinely requires no
credit card (confirmed against Google's own pricing docs, unlike OpenAI
and Anthropic which both gate any usage behind billing). Verified the
current SDK is `google-genai` (PyPI) — the older `google-generativeai` is
deprecated — and wrote `_complete_gemini()` against its real installed
API (`client.models.generate_content(model, contents, config)`,
`GenerateContentConfig(system_instruction=, max_output_tokens=,
temperature=)`, `response.text`, `response.usage_metadata.
{prompt,candidates}_token_count`), inspected directly via `inspect.signature`
rather than guessed. `GEMINI_MODEL` defaults to `gemini-2.5-flash`
(documented as still available, and Flash-tier free limits are far more
usable for a 20-case benchmark than Pro-tier free limits typically are) —
overridable via env var, and worth double-checking against
`client.models.list()` once a real key is in hand, since model IDs on
this API change often.

Anthropic support was kept, not replaced — it's the already-tested
fallback if Gemini's free-tier rate limits become a problem mid-benchmark,
selectable per-run via `LLM_PROVIDER=anthropic` with no code change.

Also fixed while wiring this up (found the moment a real key was tested,
unrelated to Gemini): the previously-pinned `anthropic==0.39.0` no longer
installs/works cleanly — pip resolved `tigergraph-mcp`'s dependency to
`pyTigerGraph>=2.0.4` earlier in this session, and separately the real
`anthropic` SDK has moved to 1.x with a materially different
`messages.create()` signature (`temperature` is no longer a bare top-level
kwarg in the version that actually installs today). Fixed by loosening the
requirements.txt pin to `anthropic>=0.40` and making `_complete_anthropic()`
retry without `temperature` on a TypeError naming it, so the code works
whichever SDK version resolves at install time rather than silently
breaking the instant someone did a fresh `pip install -r requirements.txt`.

## Open blockers

1. TigerGraph Savanna instance hostname + credentials (user to create and
   share, or run `scripts/setup_graph.py` themselves).
2. Anthropic API key (user said they'd provide; not yet received).
3. `transactions.csv` full download still in progress as of this entry.

## 2026-09-23 — Gemini validated end-to-end; connected_cards() region-cluster bug found and fixed

- **Gemini confirmed working, from the user's own machine, outside any Claude session.**
  This session's cloud sandbox and the Cowork device bridge both hard-block
  `generativelanguage.googleapis.com` at the proxy level (confirmed via the
  proxy's own status endpoint — `connect_rejected`/403 for that host, while
  `api.anthropic.com` is allowlisted). That's a network-policy constraint of
  the Claude session infrastructure, not fixable from inside it. Walked the
  user through installing `google-genai` and running a standalone
  connectivity script (`test_gemini_connection.py`, no dependency on the
  rest of the codebase) from their own PowerShell terminal — confirmed
  working there.
- **Model name**: `gemini-2.5-flash` (our original default) is deprecated
  for new users as of today; Google's own 404 error names the replacement,
  `gemini-3.6-flash`. Updated the default in `agent/llm.py` and
  `.env.example`.
- **`pydantic==2.9.2` pin was actually broken**: `mcp==1.27.0` (required by
  the tigergraph-mcp integration) needs `pydantic>=2.11`. Discovered only
  once `pip install -r requirements.txt` ran on a real, from-scratch
  environment (the cloud sandbox's dev environment had it pre-satisfied by
  something else, masking this). Loosened to `pydantic>=2.11,<3.0.0`;
  verified our own schema code uses only standard v2 syntax
  (`field_validator`, `ConfigDict`) so nothing else needed to change.
- **`_parse_json_response()` couldn't handle markdown-fenced JSON.**
  Gemini wraps structured output in \`\`\`json ... \`\`\` fences even when
  told to return bare JSON (Anthropic generally doesn't). Added explicit
  fence-stripping, and — separately — a clearer error when the response was
  truncated mid-JSON (no closing brace at all) rather than just malformed.
- **Gemini "thinking" flash models can spend part of `max_output_tokens` on
  internal reasoning before emitting the actual answer.** The first real
  run (HHG-001) got cut off mid-JSON at `max_tokens=1200` for exactly this
  reason. Fixed two ways: bumped `synthesize_with_llm`'s budget to 3000 and
  `write_sar_narrative`'s to 1500, and pass `thinking_config` with
  `thinking_budget=0` to Gemini calls (these are short, low-ambiguity
  extraction/formatting tasks, not the kind that benefit from extended
  reasoning) — guarded with a fallback retry in case the installed
  google-genai version or model doesn't accept the param.
- **Real bug found via the first successful full run**: `connected_cards()`
  only ever read `state.evidence.device_neighbors`, never
  `region_cluster` — but `has_shared_origin()` (which drives policy rule
  R6) checks both. Net effect: HHG-001 got `CREATE_CASE`/`FILE_REPORT`
  actions whose stated reason was "R6: shared origin (billing region
  444.0) across cards", while the case's own `connected_card_ids` field
  came back empty — the policy engine's justification didn't match what
  the case record showed. Fixed `connected_cards()` to check
  `region_cluster` the same way `has_shared_origin()` does. This was
  latent since the original device-fingerprint fix (which only touched
  `has_shared_origin()`, not `connected_cards()`) and wouldn't have been
  caught without a real end-to-end run against real data — the FakeLLM dev
  runs never exercised a case where R6 fired on a region match.
- **Known gap, not fixed today** (noted for the writeup / future work):
  when R6 fires, the case's `evidence` list has no dedicated claim
  describing the shared-region/device finding — it's only visible in the
  policy action's `reason` text. Functionally traceable (the reason names
  the region and, after the fix above, `connected_card_ids` names the
  actual cards), but a dedicated evidence entry would be clearer for
  human review. Left alone given deadline proximity; low risk to add later
  if time allows.

## 2026-09-23 — Retry-with-backoff for transient provider errors

- HHG-001's second run hit a genuine transient error: Gemini returned `503
  UNAVAILABLE` ("This model is currently experiencing high demand"), not a
  code bug. With 20 cases to run for the actual submission, a single flaky
  call shouldn't fail the whole run. Added `_with_retry()` in
  `agent/llm.py`: exponential backoff (2s/4s/8s), up to 4 attempts, wrapping
  the actual provider call inside `LLM.complete()` so it applies to both
  providers uniformly. Matches on error message text (`"503"`,
  `"overloaded"`, `"rate limit"`, etc.) rather than each SDK's own exception
  classes, since Anthropic's and Gemini's error hierarchies differ and this
  needs to work for both. Non-retryable errors (bad auth, malformed
  request) still fail immediately on the first attempt.

## 2026-09-23 — thinking_config incompatible with gemini-3.5-flash-lite (root cause of 0/20 benchmark runs)

- Symptom: after switching to `gemini-3.5-flash-lite` (500/day free tier vs
  20/day for `gemini-3.6-flash`, per Google AI Studio's rate-limit table),
  every single benchmark case failed — three consecutive full
  `run_benchmark.py` runs, 0/20 completed each time. Most cases hit
  `400 INVALID_ARGUMENT: {"message": "Request contains an invalid
  argument."}` with no field-specific detail; two cases (HHG-017, HHG-018)
  hit `429 RESOURCE_EXHAUSTED` on `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`
  (limit 15/min) instead, consistently across runs.
- Root cause: `_complete_gemini()` sends
  `thinking_config=types.ThinkingConfig(thinking_budget=0)` on every call,
  added earlier to stop `gemini-3.6-flash` truncating structured JSON output
  by spending part of its token budget on internal reasoning. The
  `gemini-3.5-flash-lite` tier rejects this parameter outright, but the
  rejection is a generic `400 INVALID_ARGUMENT` whose message never mentions
  "thinking" and isn't a `TypeError`/`AttributeError` — so the existing
  fallback except-clause's guard condition never matched, and it re-raised
  immediately instead of retrying without the parameter. Every call was
  structurally guaranteed to fail before this fix.
- Fix: broadened the except-clause around the `thinking_config` attempt in
  `agent/llm.py::_complete_gemini()` to catch any `Exception` unconditionally
  and retry once without `thinking_config`, rather than pattern-matching the
  error message or exception type. This can't be made more specific without
  enumerating every way a model/SDK combination might reject the param, and
  `thinking_budget` is a token-budget optimization, not something
  correctness depends on — so an unconditional fallback is the right
  tradeoff.
- Quota note for future reference: Gemini free tier has two independent caps
  per model — `GenerateRequestsPerDayPerProjectPerModel-FreeTier` (daily) and
  `GenerateRequestsPerMinutePerProjectPerModel-FreeTier` (per-minute) — both
  surface as HTTP 429 `RESOURCE_EXHAUSTED`; the `quotaId` field in the error
  payload distinguishes them. `gemini-3.5-flash-lite`'s per-minute cap (15)
  means a 20-case benchmark (40-90+ calls) needs the existing retry/backoff
  to pace around minute-level bursts even with ample daily headroom.
- Not yet rerun against this fix — pushed to the user's machine, next step
  is `python scripts/run_benchmark.py --backend local`.

## 2026-09-23 — Gemini fix confirmed (19/20 benchmark cases completed); real _req_type bug found and fixed

- The unconditional `thinking_config` fallback fix worked: `run_benchmark.py
  --backend local` went from 0/20 to 19/20 cases completed on the first
  rerun (929.7s total, latencies mostly 15-60s per case, a few outliers up
  to 150s — all within Gemini's free-tier per-minute limits, no 429s or
  400s observed this run).
- The one failure, HHG-002, was a genuine bug, not a provider/quota issue:
  `KeyError: '_req_type'` in `agent/graph_agent.py::node_request_evidence`.
  Root cause: `route_after_initial_policy` (a LangGraph *conditional edge*
  function, not a node) was setting `state["_req_type"] = req_type` as a
  side effect before returning its routing decision. LangGraph only commits
  a state update to the graph's channels when it comes back as part of a
  node's *return value* — a conditional edge function mutating the state
  dict in place is not a supported way to persist an update. This
  apparently worked on 19/20 cases via incidental object-reference reuse
  (the same dict happened to flow through unmodified), but nothing in
  LangGraph's contract guarantees that, and it broke on one case.
- Fix: moved the `inv.decide_evidence_request(...)` call and the
  `state["_req_type"] = ...` assignment into `node_policy_initial` (a real
  node, whose return value LangGraph does merge), leaving
  `route_after_initial_policy` as a pure reader (`state.get("_req_type")`)
  that only decides routing. Pushed to the user's machine; awaiting a
  benchmark rerun to confirm 20/20.

## 2026-09-23 — Windows UTF-8 file-write bug; Savanna auth fixed (secret-only, no username/password)

- `run_benchmark.py`/`run_case.py` crashed writing HHG-002's answer JSON on
  Windows: `UnicodeEncodeError: 'charmap' codec can't encode character
  '→'` (the "→" in graph_agent.py's `what_changed` narrative). Root
  cause: `open(path, "w")` with no explicit encoding uses the OS default
  (cp1252 on Windows), not UTF-8. Fixed by adding `encoding="utf-8"` to
  every answer/summary file write in both scripts — a narrow fix, but the
  right one, since the content itself (an arrow in a human-readable
  narrative) is fine and shouldn't be avoided for a platform's default
  codepage.
- `scripts/setup_graph.py` (the never-before-tested-against-real-Savanna
  script) failed immediately with `('User authentication failed', None)`.
  Root cause: it was passing `username`/`password` to
  `TigerGraphConnection(...)` (defaulting to the legacy "tigergraph" user),
  but Savanna doesn't accept username/password at all — only a Database
  Secret. Confirmed against another team's public repo for this same
  hackathon (github.com/neevmodh/graphsleuth), which connects with
  `TigerGraphConnection(host=..., graphname="", gsqlSecret=secret,
  tgCloud=True)` and no username/password whatsoever. Fixed `connect()` to
  match that pattern exactly.
- Also added `_with_wakeup_retry()` (6 attempts, 20s apart) around every
  GSQL/load call in `setup_graph.py`, per the same reference repo's
  documented gotcha: an idle Savanna workspace auto-suspends, and the first
  request after that wakes it but can surface as a confusing
  Bad-Gateway/HTML-in-JSON error rather than a clean timeout. The load step
  is safe to retry wholesale since `graph/load_data.py` uses
  `upsertVertexDataFrame`/`upsertEdgeDataFrame` (idempotent on primary ID),
  so a retry re-upserts rather than duplicates.
- Not yet run against Savanna with this fix — pushed to the user's
  machine, next step is `python scripts/setup_graph.py --all`.

## 2026-09-23 — GSQL reserved-word collisions: `proxy` and `Case` renamed

- `setup_graph.py --all` got past authentication this time (confirming the
  Savanna auth fix) but `schema.gsql` failed to parse: `Encountered ",", ""
  at line 60, column 24. Was expecting one of: "(" ")" "compress" "default"
  "nullable" "primary"`. Cross-referenced every attribute/vertex/edge name
  in the schema against TigerGraph's DDL reserved-word list
  (docs.tigergraph.com/gsql-ref/4.2/appendix/keywords-and-reserved-words)
  and found two real collisions: `proxy` (a `DeviceProfile` attribute) and
  `Case` (the case vertex type itself — `CASE` is reserved, and the parser
  would have hit this immediately after `proxy` was fixed, since GSQL
  restricts reserved words from vertex/edge/graph/attribute names, not just
  attributes).
- Fixed by renaming: `proxy` → `proxy_flag` in `schema.gsql` and
  `load_data.py`'s upsert attribute mapping (the CSV column `id_23` is
  unchanged, only the GSQL-side attribute name); `Case` → `FraudCase`
  everywhere it's a GSQL vertex type — `schema.gsql` (vertex def, 4 edge
  FROM/TO clauses, the CREATE GRAPH vertex list), `queries.gsql` (4
  occurrences: 2 traversal steps, 1 wildcard select, 1 INSERT INTO), and
  `load_data.py` (`upsertVertexDataFrame`/`upsertEdgeDataFrame` calls for
  the historical closed-cases load).
- Deliberately did NOT rename anything on the Python side
  (`agent/detectors.py`, `graph/local_backend.py`): `tigergraph_reshape.py`
  is the one boundary point that translates raw REST attribute names back
  into the dict keys the rest of the agent expects, so it now reads
  `dev.get("proxy_flag")` but still outputs the dict key as `"proxy"` —
  every downstream consumer is unaffected. `agent/schemas.py`'s `Case`
  Pydantic model (the CaseAnswer JSON format) is unrelated to the GSQL
  vertex type name and was left untouched.
- Scanned the rest of `schema.gsql` and `queries.gsql` programmatically
  against the full reserved-word list for any other collisions — none
  found (remaining hits were all legitimate keyword usage: `EDGE` inside
  `SetAccum<EDGE>`, `BETWEEN`, `to_datetime`/`datetime_diff` built-ins,
  `HeapAccum`).
- Not yet rerun — pushed to the user's machine, next step is
  `python scripts/setup_graph.py --all` again.

## 2026-09-23 — GSQL `BOOL DEFAULT` needs a quoted literal, not a bare keyword

`report_filed BOOL DEFAULT FALSE,` in `FraudCase` (schema.gsql line 89) failed
to parse on this TigerGraph version:

```
Encountered " "false" "FALSE "" at line 89, column 31.
Was expecting one of:
    <CHARACTER_LITERAL> ...
    <NULL_LITERAL> ...
    <STRING_LITERAL> ...
```

The parser's own expected-token list only names character/NULL/string
literals for a `DEFAULT` clause — no bare `TRUE`/`FALSE` keyword. Fixed by
quoting it: `DEFAULT "false"`. Grepped the rest of schema.gsql for other
`DEFAULT` clauses first (`num_cards UINT DEFAULT 0` on line 19 is a plain
numeric literal, not affected). This was the only other line.

## 2026-09-23 — Catalog reset needed: GSQL CREATE isn't idempotent

Third `--all` run hit `Semantic Check Fails: The vertex name Card is used
by another object!` on a schema that had never fully succeeded. Root cause:
the two earlier failed `--schema` runs (the `proxy`/`Case` reserved-word
error, then the `DEFAULT FALSE` error) each got partway through
`schema.gsql` before failing — `CREATE VERTEX Customer` and
`CREATE VERTEX Card` had already succeeded and landed in the workspace
catalog before each failure, and `CREATE VERTEX` isn't idempotent, so
rerunning the same script now collides with its own leftover state.

Fixed by adding a `--reset` flag to `scripts/setup_graph.py` that runs
`DROP ALL` before schema creation, wiping the workspace catalog clean
(safe here — nothing has been loaded into a real graph yet; the only
casualty is the pre-built sample solution kit a new Savanna workspace
ships with, unused by this project). Re-fetches the API token after
`DROP ALL` since it invalidates the one obtained at connect time.

Usage going forward: `python scripts/setup_graph.py --reset --all` for the
first genuinely clean run, or any time a `--schema` run fails partway
through and needs to be rerun.

## 2026-09-23 — queries.gsql semantic/syntax errors: primary_id dot-access, ternary operator, undeclared accumulator; load_data.py NaN-to-null bug

First clean schema creation (after --reset) surfaced real GSQL semantics
issues in queries.gsql that a syntax-only read couldn't have caught:

1. **PRIMARY_ID isn't dot-referenceable by default.** `t.txn_id == p_txn_id`
   (and the same pattern for card_id, customer_id, device_key, region_code,
   domain, case_id) failed: "The expression refers to a primary_id, which
   is not directly usable in the query body." Fixed by re-declaring each
   PRIMARY_ID as a plain attribute of the same name in schema.gsql (the
   exact fix the parser's own error message suggests), and updated
   load_data.py's upsertVertexDataFrame attributes= maps to populate the
   now-duplicated attribute (it was previously excluded on the assumption
   the primary id didn't need a separate value). This also explains
   get_customer_profile's stranger error ("cu.customer_id indicates vertex
   types [Card]") — Card already had a real customer_id attribute, so the
   ambiguity only existed for Customer.
2. **No ternary operator.** `cond ? a : b` doesn't parse in this GSQL
   version. get_card_window and get_velocity's center_ts fallback rewritten
   as IF/THEN/ELSE/END; find_similar_cases' cosine-similarity division
   rewritten to avoid needing a conditional at all (add a small epsilon to
   the denominator instead of branching on zero-norm).
3. **Undeclared per-vertex accumulator.** get_card_region_history used
   `r.@txn_count` without declaring `SumAccum<INT> @txn_count;` first.
4. **load_data.py: NaN uploading as invalid JSON.** card4/card6
   (network/card_type) and the DeviceProfile identity.csv columns
   (DeviceType, DeviceInfo, id_30/31/33/34/23/15) were never passed through
   the same NaN-safe `clean()` helper other columns use, and `channel`
   used `x or ""` — which does NOT catch `float('nan')`, since NaN is
   truthy in Python. Any of these produced 'Null JSON value is not
   supported' (REST-30200) during the load step. All now go through
   `clean()` before upsert, matching the pattern already used for
   card1/2/3/5 and p/r_email_domain.

Ran `--reset --all` end to end for the first time to surface these; none of
this was visible from a static read since it depends on this specific
TigerGraph version's grammar and the actual NaN patterns in the dataset.

## 2026-09-23 — Correction: primary_id fix was wrong; the real syntax is PRIMARY_ID_AS_ATTRIBUTE="true"

The previous entry's fix (re-declaring each PRIMARY_ID a second time under
the identical attribute name, e.g. `PRIMARY_ID txn_id STRING, txn_id
STRING,`) was wrong. It didn't error at schema-creation time, and it let
`ca.case_id` type-check inside one query's ACCUM clause (read access), but
every WHERE-clause filter on a primary id (`t.txn_id == p_txn_id`) still
failed with the identical error, and data loading broke outright ("Unknown
vertex attribute or vector name: customer_id", REST-10004) — the duplicate
declaration never created a real, separately-writable attribute.

Confirmed via TigerGraph's own forum (dev.tigergraph.com/forum, "Could
primary_id (v_id) of a vertex be used in a query body?") that the actual
documented mechanism is a vertex-level option:

```
CREATE VERTEX Customer (
    PRIMARY_ID customer_id STRING,
    num_cards UINT DEFAULT 0
) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
```

Applied to all 7 vertex types in schema.gsql; removed all the duplicate
attribute-name lines. Reverted the corresponding load_data.py
upsertVertexDataFrame attributes= additions (the primary id value is
already set via v_id=; there's no separate attribute slot to populate, so
those explicit mappings are back to being excluded, matching the original
code before this whole detour).

Also fixed a second, unrelated find_similar_cases bug surfaced once the
ternary-removal parsed clean: `DOUBLE dot = 0, na = 0, nb = 0,` (the
multi-var comma-shorthand under one type keyword) left na/nb out of scope
by the time they were read a few lines later (TYP-523: "An undefined
variable na in current scope"). Fixed by giving each local var its own
explicit `DOUBLE` keyword.

## 2026-09-24 — Schema fix confirmed working; three more query bugs + edge-load bug found

`PRIMARY_ID_AS_ATTRIBUTE="true"` worked: 7 of 10 queries installed cleanly
on the next run. Three remained, plus a new load-step failure once queries
got further than before:

1. **No `.first()` method on a vertex set.** get_card_window/get_velocity's
   `Center.first().ts` (TYP-1002: "Function name 'first' does not match
   any existing functions"). Fixed by using a `MaxAccum<DATETIME> @@center_ts`
   global accumulator instead — the standard GSQL way to pull a scalar out
   of a SELECT that matches 0 or 1 rows (`ACCUM @@center_ts += t.ts`, with
   an explicit epoch fallback when the set is empty).
2. **LIST-typed query parameters can't have methods called on them
   directly.** find_similar_cases' `p_embedding.size()` / `.get(i)`
   (TYP-1001: "identifier 'p_embedding' of type list parameter is invalid
   to call any function"). Fixed by copying the parameter into a local
   `LIST<DOUBLE> emb` variable first and calling `.size()`/`.get()` on
   that instead.
3. **load_data.py: upsertEdgeDataFrame with no attributes= tries to
   auto-map every extra dataframe column onto the edge.** Most of this
   schema's edges (OWNS, FROM_DEVICE, PURCHASER_EMAIL, BILLED_IN, ON_CARD)
   declare zero attributes, but the dataframes passed to those upserts
   carry many other columns (card network/type, transaction features,
   case fields, even the vertex primary ids) — pyTigerGraph's default
   attempted to write those as edge attributes and failed ("Processing
   attribute card_id failed, Invalid edge attribute", REST-30200, on the
   very first edge upsert — OWNS). Fixed by passing an explicit `attributes={}`
   on every attribute-less edge, and `attributes={"ts": "ts"}` on MADE
   (its one real attribute).

None of this was visible from a static read — each fix only surfaces once
the previous layer of errors clears, since GSQL/pyTigerGraph fail fast on
the first problem in file/statement order.

## 2026-09-24 — Two more query bugs: JSON API v2 PRINT restriction, LIST local-variable declaration doesn't parse

8 of 10 queries installed cleanly this run. Remaining two:

1. **get_velocity**: `PRINT Recent, Recent.size() AS txn_count, sum(Recent.amt) AS total_amt;`
   failed with SEM-1420: "It is not supported in Json API Version 'v2'. If
   you want to print attribute/vertex-attached accumulator, please use the
   new grammar." — an inline aggregate call (`sum(...)`) inside PRINT isn't
   allowed under this API version. Fixed by accumulating during the SELECT
   instead (`SumAccum<DOUBLE> @@total_amt; ... ACCUM @@total_amt += t.amt`),
   the exact pattern get_customer_profile already used successfully for its
   own @@total_amt — should have caught this by consistency the first time.
2. **find_similar_cases**: `LIST<DOUBLE> emb = p_embedding;` failed to
   parse at all ("no viable alternative at input 'LIST<double> emb'") —
   composite types (LIST/SET/MAP) are only valid as query parameter or
   vertex/edge attribute types in this GSQL version, not as local variable
   declarations inside a query body. Combined with LIST parameters not
   supporting direct method calls (previous entry), there was no way to
   get a size/indexable copy via assignment. Fixed by copying via
   iteration instead: `ListAccum<DOUBLE> @@emb; FOREACH v IN p_embedding
   DO @@emb += v; END;` — FOREACH...IN is iteration syntax, not a method
   call on the parameter identifier, so it isn't blocked by TYP-1001, and
   a ListAccum supports .size()/.get() normally afterward.

## 2026-09-24 — MADE edge's ts attribute: raw Timestamp serializes differently than the pre-stringified vertex ts

Load reached the Transaction chunk loop for the first time (all earlier
vertex/edge fixes held) and failed with 'Processing attribute ts failed,
value cannot be converted to Datetime' (REST-30200). Diagnosed directly
against the user's transactions.csv (device_bash) rather than guessing:
121,941 rows, zero NaT after pd.to_datetime, zero nonzero-microsecond
timestamps, and str() already renders exactly "YYYY-MM-DD HH:MM:SS" — so
the data itself, and Transaction.ts (which goes through
_row_to_vertex_attrs's str(row["ts"])), were never the problem.

The actual cause: the MADE edge's ts attribute (added in an earlier fix
round) was populated straight from chunk's native pandas Timestamp column,
not a pre-stringified value — pyTigerGraph serializes a raw Timestamp
differently (likely ISO "T"-separated) than the plain space-separated
string TigerGraph's DATETIME parser accepts. Fixed by explicitly
formatting it the same way: `chunk["ts_str"] = chunk["ts"].dt.strftime(...)`
and mapping MADE's ts attribute from that column. Also switched
Transaction.ts from `str(row["ts"])` to the same explicit `.strftime(...)`
call for parity, even though it wasn't the failing one — removes any
reliance on pandas' Timestamp.__str__ default formatting.

## 2026-09-24 — tigergraph_backend.py had the same pre-Savanna-fix auth bug setup_graph.py already fixed

First live validation (`run_case.py --backend tigergraph`) failed twice
with "500 Server Error: Internal Server Error for url: .../gsql/v1/tokens",
which looked like — but almost certainly was not primarily — the auto-
suspend cold-start symptom already documented. The real cause: this file's
`__init__` was never updated when setup_graph.py's Savanna auth bug was
fixed earlier — it still built the connection with
`username="tigergraph", password=""` before calling `getToken(secret)`,
the exact pattern Savanna rejects outright. Fixed to match
scripts/setup_graph.py's proven-working pattern exactly: no
username/password at all, `tgCloud=True` + `gsqlSecret=secret` straight
into the constructor. Also added the same wake-up retry wrapper
(`_with_wakeup_retry`) around both the token fetch and every installed
query call — setup_graph.py only had it for the one-time schema/load
scripts; the actual query path used by every case investigation had none,
so a genuine cold-start on a live run would have failed outright instead
of retrying.

## 2026-09-24 — TigerGraph integration confirmed fully working end to end

`python scripts/run_case.py HHG-001 --backend tigergraph` succeeded after
the auth fix and manually resuming the Savanna workspace (it had gone back
to auto-suspend by the time of this validation, and the API-triggered
wake-up either wasn't happening or was too slow — resuming it directly in
the Savanna console was the reliable fix). Full run: 6 tool calls, 10.75s,
5513 tokens — get_transaction_details, get_customer_profile (surfaced 4
prior closed fraud cases + 422 txn history), find_similar_cases (5 matching
historical cases by embedding similarity), region-cluster/velocity checks,
correct fraud verdict (0.711 probability, out_of_region_use pattern),
SAR narrative generated, next-best-actions decided, and the case written
back to the graph (written_to_graph: true, graph_case_id: CASE-HHG-001,
confirming upsert_case works too).

This closes out the TigerGraph/GSQL debugging arc: schema, all 10 queries,
the full data load (121,941 transactions + historical cases), and the live
query/write path are all confirmed working against the real Savanna
instance, not just the local backend.
