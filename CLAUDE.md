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

# Tests (scripted fake LLM, no provider calls)
pytest
pytest tests/test_graph.py::test_step_cap_forces_tool_free_summary   # single test
```

## Configuration

Settings come from `.env` via `agent/config.py` (Pydantic Settings). Key vars: `LLM_PROVIDER` (`openai` | `lmstudio`), `LLM_MODEL` (default `gpt-4o-mini`), `LLM_EXPLORE_MODEL` (optional cheaper model for `explore` subagents; falls back to `LLM_MODEL` — leave unset for LM Studio), `LLM_TEMPERATURE`, `OPENAI_API_KEY`, `LMSTUDIO_BASE_URL`. The OpenAI/LM Studio switch is centralized in `agent/llm.py::get_llm(model=None)` — both providers are served through `ChatOpenAI`, so LM Studio is just OpenAI with a different `base_url`.

## Architecture

Two top-level packages live under `src/` (`pyproject.toml` sets `packages.find where=["src"]`), so imports are `from agent...` and `from tools...` — **not** `from src.agent...`.

The agent is a hand-built ReAct loop. `agent/graph.py::build_graph(llm, tools, hitl, max_steps=None, subagents=None, checkpointer=None)` is a factory; the parent agent and both subagent types are three configurations of the same loop, assembled in `agent/subagents.py::build_agent()` (what the CLI runs).

- `State` is `{messages: Annotated[list, add_messages], todos, steps, description}` — `add_messages` appends; `steps`/`description` are only used by subagents.
- `chat_node` → LLM (tools bound) produces a message; increments `steps`. Past `max_steps` it makes one tool-free call with `STEP_LIMIT_PROMPT` and returns the summary.
- `route_after_chat` → no `tool_calls` → `END`; otherwise `human_approval` (if `hitl`) or straight to `dispatch`.
- `human_approval_node` → one `interrupt()` per tool call unless `config.configurable.auto_approve`; rejections become `ToolMessage`s so the model sees why.
- `dispatch` → approved `task` calls become `Send("subagent", tool_call)` (one node instance each, run in parallel in the same superstep); any other approved call adds `"tools"`; everything rejected → `"chat"`.
- `tool_node` → runs ordinary calls from `TOOLS_BY_NAME`, skips `task` and already-answered (rejected) calls.
- `subagent_node` → picks the compiled subgraph by `subagent_type`, seeds `SystemMessage(prompt) + HumanMessage(task.prompt)`, invokes it **with the node's own `config`** (this is what makes the subgraph inherit the checkpointer and lets its `interrupt()` bubble up as `__interrupt__` on the parent stream — a hand-built config would silently swallow it), returns one `ToolMessage`. Exceptions become `"Subagent failed: …"`; `GraphBubbleUp` is re-raised.

Subagents (`agent/subagents.py`): `explore` = read-only tools + `web_search`, `hitl=False`, `get_llm(cfg.llm_explore_model)`; `general` = read-only + `edit_file`/`write_file`, `hitl=True` (inherits `/auto`). Both `max_steps=20`, neither has `task` (depth 1) or the todo tools. Prompts in `agent/prompts/{explore,general}.md`; `system.md` has the parent's delegation guidance.

`task` (`tools/task_tool.py`) is a **Pydantic schema, not a function** — bound to the parent LLM via `bind_tools`, never executed by `tool_node`. Its class docstring is the LLM-facing spec.

Adding an ordinary tool: define it with `@tool` in `tools/`, then add it to the right list in `subagents.py` (`READ_ONLY_TOOLS` / `WRITE_TOOLS` / `PLANNING_TOOLS`). The docstring is the LLM's spec — see existing tools for the house style: when-to-use guidance, examples, truncation notes, actionable error strings.

`agent/cli.py` is the REPL. `run_turn` streams with `stream_mode=["updates", "custom"]` and `subgraphs=True`, so events are `(namespace, mode, payload)`; root events have `namespace == ()`, a subagent's are `("subagent:<task_id>",)`. The subgraph emits `{"description": …}` on the `custom` channel at its first `chat` step; the CLI maps namespace → label and renders inner tool calls as `[label] [tool] …` while hiding inner tool output. `task` results are rendered in full; other tool output is truncated to 400 chars. All `__interrupt__` events in a step are collected and resumed together with `Command(resume={interrupt.id: decision})` (mandatory when several are pending). Completed nodes in an interrupted superstep have their writes replayed on resume — `render_update` dedupes `ToolMessage`s by `tool_call_id` so nothing prints twice. The system prompt is injected only on the first turn.

Tools (`tools/`): `list_dir`, `read_file`, `glob`, `grep`, `edit_file`, `write_file` (`file_tools.py`), `web_search` (`research_tools.py`, DuckDuckGo via `ddgs`), `write_todos`/`read_todos` (`todo_tools.py`, `InjectedState`), `task` (`task_tool.py`).

Tests live in `tests/test_graph.py` and drive the graph with a scripted `BaseChatModel` (`ScriptedChat`) — a `respond(messages, tools_bound)` callable, so parallel subagents can share one model deterministically. Nothing hits a provider.
