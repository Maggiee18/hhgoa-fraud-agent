"""
GraphClient backend that talks to TigerGraph through the TigerGraph MCP
server (https://pypi.org/project/tigergraph-mcp/) instead of a direct
pyTigerGraph connection — this is what satisfies the challenge's required
component "Use TigerGraph MCP to expose graph capabilities and data to the
agent" literally: every graph read/write the agent makes goes through the
MCP protocol.

Subclasses graph.tigergraph_backend.TigerGraphClient and overrides only
__init__ and _run_installed_query, so every query name, parameter dict, and
result reshape is identical to (and can never drift from) the direct
pyTigerGraph backend — see that module's docstring. Confirmed against the
real `tigergraph-mcp` package (pip installed and inspected directly, and
exercised live over stdio against a dummy TG_HOST — see docs/DECISIONS.md
for what that run did and didn't prove) that
`tigergraph__run_installed_query` takes {query_name, params, profile?,
graph_name?} and internally calls conn.runInstalledQuery(query_name,
params), and that its response text is NOT plain JSON — it's a fenced
```json ... ``` block followed by a human-readable rendering of the same
payload (see _parse_tool_text below), for both the success and error
shape.

Transport, from env:
  TG_MCP_URL set        -> streamable-http, to that URL's /mcp/ endpoint
                            (TG_MCP_PROFILE/TG_USERNAME+TG_PASSWORD/
                            TG_API_TOKEN travel as X-TG-* headers)
  TG_MCP_URL unset       -> stdio: this process spawns `tigergraph-mcp`
                            itself, passing TG_* as its env (stdio_client
                            does not inherit the parent's env — confirmed
                            from the package's own README)

Async bridging: the mcp SDK's ClientSession and its transport are async
context managers whose cancel scopes must be entered AND exited from the
same asyncio Task (anyio requirement) — naively calling __aenter__/__aexit__
from separate run_coroutine_threadsafe submissions breaks that and raises
"Attempted to exit cancel scope in a different task" (hit and fixed during
this build — see docs/DECISIONS.md). _SessionWorker instead runs ONE
long-lived coroutine that opens the session, then loops pulling
(future, coro_fn) work items off an asyncio.Queue fed from other threads,
so the whole session lifetime — connect, every call, disconnect — happens
inside a single task.

UNTESTED end-to-end against a live TigerGraph instance behind the MCP
server (no TigerGraph credentials available in this environment); the
transport/session bootstrap and error-shape parsing below WERE exercised
live against the real tigergraph-mcp package over stdio with a
deliberately-unreachable TG_HOST, which is what surfaced and let us fix
the two bugs described above.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import threading
from typing import Any, Callable, Coroutine, Dict, List, Optional

from dotenv import load_dotenv

from graph.tigergraph_backend import TigerGraphClient
from graph import tigergraph_reshape as reshape

load_dotenv()


class _SessionWorker:
    """Owns one MCP ClientSession for its whole lifetime inside a single
    background asyncio task, so all of its async-context-manager cancel
    scopes are entered and exited from that same task (see module
    docstring). Synchronous callers submit `session -> awaitable` functions
    via `call()` and block for the result."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._queue: Optional[asyncio.Queue] = None
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._main_future: Optional[concurrent.futures.Future] = None

    def start(self, session_cm_factory: Callable[[], Any], timeout: float = 60):
        """session_cm_factory() must return an async context manager
        yielding a live mcp.ClientSession (already .initialize()'d)."""
        self._main_future = asyncio.run_coroutine_threadsafe(self._main(session_cm_factory), self._loop)
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("Timed out connecting to TigerGraph MCP server")
        if self._init_error is not None:
            raise self._init_error

    async def _main(self, session_cm_factory):
        self._queue = asyncio.Queue()
        try:
            async with session_cm_factory() as session:
                await session.initialize()
                self._ready.set()
                while True:
                    item = await self._queue.get()
                    if item is None:
                        break
                    fut, coro_fn = item
                    try:
                        result = await coro_fn(session)
                        if not fut.cancelled():
                            fut.set_result(result)
                    except BaseException as e:  # noqa: BLE001 - relayed to the caller's thread
                        if not fut.cancelled():
                            fut.set_exception(e)
        except BaseException as e:  # connection/handshake failure
            self._init_error = e
            self._ready.set()

    def call(self, coro_fn: Callable[[Any], Coroutine], timeout: float = 120) -> Any:
        if self._queue is None:
            raise RuntimeError("TigerGraph MCP session was not started")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (fut, coro_fn))
        return fut.result(timeout=timeout)

    def close(self, timeout: float = 30):
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            if self._main_future is not None:
                try:
                    self._main_future.result(timeout=timeout)
                except Exception:
                    pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def _session_cm_factory():
    """Builds the async context manager for one MCP session: (transport) ->
    ClientSession, chosen by transport per the module docstring. Returned as
    a zero-arg callable so _SessionWorker can `async with factory():`."""
    from contextlib import asynccontextmanager

    from mcp import ClientSession

    mcp_url = os.getenv("TG_MCP_URL", "").strip()

    @asynccontextmanager
    async def _http():
        from mcp.client.streamable_http import streamablehttp_client

        headers: Dict[str, str] = {}
        if os.getenv("TG_MCP_PROFILE"):
            headers["X-TG-Profile"] = os.environ["TG_MCP_PROFILE"]
        if os.getenv("TG_API_TOKEN"):
            headers["X-TG-Api-Token"] = os.environ["TG_API_TOKEN"]
        elif os.getenv("TG_USERNAME") and os.getenv("TG_PASSWORD"):
            headers["X-TG-Username"] = os.environ["TG_USERNAME"]
            headers["X-TG-Password"] = os.environ["TG_PASSWORD"]
        if os.getenv("TG_HOST"):
            headers["X-TG-Host"] = os.environ["TG_HOST"]
        if os.getenv("TG_GRAPH_NAME"):
            headers["X-TG-Graphname"] = os.environ["TG_GRAPH_NAME"]

        url = mcp_url.rstrip("/") + "/mcp/"
        async with streamablehttp_client(url, headers=headers or None) as (read, write, _close):
            async with ClientSession(read, write) as session:
                yield session

    @asynccontextmanager
    async def _stdio():
        from mcp import StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        env = {
            **get_default_environment(),
            "TG_HOST": os.environ.get("TG_HOST", ""),
            "TG_GRAPHNAME": os.environ.get("TG_GRAPH_NAME", "HHGOA_Fraud"),
            "TG_USERNAME": os.environ.get("TG_USERNAME", ""),
            "TG_PASSWORD": os.environ.get("TG_PASSWORD", ""),
            "TG_SECRET": os.environ.get("TG_SECRET", ""),
            "TG_API_TOKEN": os.environ.get("TG_API_TOKEN", ""),
        }
        params = StdioServerParameters(command="tigergraph-mcp", args=[], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session

    return _http() if mcp_url else _stdio()


_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_tool_text(text: str) -> Dict[str, Any]:
    """tigergraph-mcp's tool responses are not plain JSON: they're a
    ```json ... ``` fenced block carrying the structured payload, followed
    by a human-readable rendering of the same data (confirmed live — see
    module docstring). Extract the fenced block first; fall back to
    brace-matching the first {...} run for robustness if a future server
    version drops the fence."""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    m2 = re.search(r"\{.*\}", text, re.DOTALL)
    if not m2:
        raise ValueError(f"tigergraph-mcp response did not contain JSON: {text[:200]}")
    return json.loads(m2.group(0))


def _unwrap_tool_result(result, operation: str) -> List[Dict[str, Any]]:
    """Unwraps an MCP CallToolResult from tigergraph-mcp's own
    format_success/format_error envelope down to the raw pyTigerGraph query
    result list that graph/tigergraph_reshape.py expects — the same shape
    TigerGraphClient._run_installed_query returns directly."""
    payload: Optional[Dict[str, Any]] = None
    if getattr(result, "structuredContent", None):
        payload = result.structuredContent
    elif result.content:
        first = result.content[0]
        text = getattr(first, "text", None)
        if text:
            payload = _parse_tool_text(text)

    if payload is None:
        raise RuntimeError(f"tigergraph-mcp {operation}: empty/unparseable tool response")
    if getattr(result, "isError", False) or payload.get("success") is False:
        raise RuntimeError(f"tigergraph-mcp {operation} failed: {payload.get('error', payload)}")

    data = payload.get("data", payload)
    return data.get("result", data)


class TigerGraphMCPClient(TigerGraphClient):
    def __init__(self):
        self._worker = _SessionWorker()
        self._worker.start(_session_cm_factory)

    def _run_installed_query(self, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        async def _call(session):
            return await session.call_tool(
                "tigergraph__run_installed_query", {"query_name": name, "params": params}
            )

        result = self._worker.call(_call, timeout=120)
        return _unwrap_tool_result(result, operation=f"run_installed_query({name})")

    def close(self):
        self._worker.close()
