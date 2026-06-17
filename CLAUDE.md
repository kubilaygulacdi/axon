# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Axon is a terminal-native coding agent built **from scratch on LangChain + LangGraph** as an educational project. The goal is not to ship the fastest agent but to understand every layer — so the codebase deliberately avoids framework "magic." `ROADMAP.md` is the living status doc (written in Turkish) and the first thing to read each session; it tracks the sprint plan and the current MVP target.

## Working rules (from ROADMAP.md — these override default instincts)

- **No black-box orchestration.** Do not use prebuilt orchestrators like `create_react_agent`. State / nodes / edges are defined by hand. Core primitives (`@tool`, `bind_tools`, `ChatOpenAI`, `StateGraph`) are fine.
- **No overengineering.** Prefer the simplest solution. Do not introduce Protocols, registries, or abstract base classes unless the simple version is genuinely insufficient.
- **`__init__.py` stays empty.** No re-exports; importers reach into submodules directly (`from tools.file_tools import grep`).
- **Minimal comments in source.** No module docstrings, no `_private` prefixes. Explanatory prose lives in `notebooks/*.md` (gitignored), not in code. Tool docstrings are the exception — they are the LLM-facing interface and should be thorough.
- **Never commit automatically.** Only commit when the user explicitly asks.
- **Branches:** only `main` (release) and `dev` (active work). No per-feature branches.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run the REPL (entry point defined in pyproject.toml [project.scripts])
axon                              # uses OpenAI by default
# /exit inside the REPL to quit

# Run against a local model via LM Studio
LLM_PROVIDER=lmstudio LLM_MODEL=qwen2.5-coder-7b-instruct axon

# Lint
ruff check src

# Tests (pytest is configured; no test files exist yet)
pytest
pytest path/to/test_x.py::test_name   # single test
```

**MLflow is a hard runtime dependency of the REPL.** `agent/graph.py` calls `mlflow.set_tracking_uri("http://localhost:5000")` and `mlflow.langchain.autolog()` at import time. An MLflow server must be reachable at `localhost:5000` or importing the graph (and thus running `axon`) will fail/hang. Start one with `mlflow server --host localhost --port 5000` before running.

## Configuration

Settings come from `.env` via `agent/config.py` (Pydantic Settings). Key vars: `LLM_PROVIDER` (`openai` | `lmstudio`), `LLM_MODEL` (default `gpt-4o-mini`), `LLM_TEMPERATURE`, `OPENAI_API_KEY`, `LMSTUDIO_BASE_URL`. The OpenAI/LM Studio switch is centralized in `agent/llm.py::get_llm()` — both providers are served through `ChatOpenAI`, so LM Studio is just OpenAI with a different `base_url`.

## Architecture

Two top-level packages live under `src/` (`pyproject.toml` sets `packages.find where=["src"]`), so imports are `from agent...` and `from tools...` — **not** `from src.agent...`.

The agent is a hand-built ReAct loop in `agent/graph.py`:

- `State` is `{messages: Annotated[list, add_messages]}` — the `add_messages` reducer appends rather than replaces.
- `chat_node` → LLM (with tools bound) produces a message.
- `route_after_chat` → if the LLM emitted `tool_calls`, go to `tools`; else `END`.
- `tool_node` → looks each call up in `TOOLS_BY_NAME`, invokes it, wraps results in `ToolMessage`s.
- Edges: `START → chat → (tools → chat)* → END`. Compiled with `InMemorySaver` checkpointer; the CLI uses a fixed `thread_id="1"`, so history persists only within one process.

Adding a tool: define it with `@tool` in `tools/`, then add it to the `TOOLS` list in `graph.py:19`. The docstring is the LLM's spec for the tool — write it carefully (see existing tools for the house style: when-to-use guidance, examples, truncation notes, actionable error strings that tell the model how to recover).

`agent/cli.py` is the REPL: streams the graph with `stream_mode="updates"` and `render_update` pretty-prints tool calls, agent text, and (truncated) tool output. The system prompt is injected only on the first turn.

Current tools (`tools/`): `list_dir`, `read_file`, `glob`, `grep`, `edit_file`, `write_file` (`file_tools.py`) and `web_search` (`research_tools.py`, DuckDuckGo via `ddgs`). File-mutating tools currently apply changes directly; per the roadmap, human-in-the-loop approval (LangGraph `interrupt()`) is the next planned milestone.
