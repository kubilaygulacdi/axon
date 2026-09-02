from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from agent.graph import TASK
from agent.subagents import build_agent, load_prompt
from tools.todo_tools import render_todos

SYSTEM_PROMPT = load_prompt("system")

TOOL_OUTPUT_LIMIT = 400


def truncate(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text)} chars total]"


def render_tool_call(tc: dict, indent: str) -> None:
    if tc["name"] == TASK:
        print(f"{indent}[task→{tc['args'].get('subagent_type')}] {tc['args'].get('description')}")
        for line in str(tc["args"].get("prompt", "")).splitlines():
            print(f"{indent}   │ {line}")
        return
    args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
    print(f"{indent}[tool] {tc['name']}({args})")


def render_update(update: dict, label: str | None = None, seen: set | None = None) -> None:
    for msg in update.get("messages", []):
        if isinstance(msg, ToolMessage) and seen is not None:
            if msg.tool_call_id in seen:      # replayed after a resume; the tool ran only once
                continue
            seen.add(msg.tool_call_id)
        if label is not None:                 # inside a subagent: show its tool calls only
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    render_tool_call(tc, indent=f"   [{label}] ")
            continue
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                if tc["name"] == "write_todos":
                    print("[plan]")
                    print(render_todos(tc["args"]["todos"]))
                    continue
                render_tool_call(tc, indent="")
            if msg.content and not msg.tool_calls:
                print("\n[agent]")
                print(msg.content)
        elif isinstance(msg, ToolMessage):
            if msg.name == "write_todos":     # already rendered as the [plan] panel
                continue
            content = msg.content if msg.name == TASK else truncate(msg.content)
            for line in content.splitlines() or [""]:
                print(f"   │ {line}")


def render_preview(call: dict) -> str:
    who = f"[{call['subagent']}] " if call.get("subagent") else ""
    lines = [f"approve> {who}{call['name']}"]
    for key, value in call["args"].items():
        text = truncate(value if isinstance(value, str) else repr(value))
        body = text.splitlines() or [""]
        if len(body) == 1:
            lines.append(f"  {key}: {body[0]}")
        else:
            lines.append(f"  {key}:")
            lines.extend(f"  │ {line}" for line in body)
    return "\n".join(lines)


def prompt_approval(call: dict) -> dict:
    print(render_preview(call))
    if input("approve? [y/N]: ").strip().lower() in ("y", "yes"):
        return {"approved": True, "reason": ""}
    reason = input("reason (sent back to the model): ").strip()
    return {"approved": False, "reason": reason or "No reason given."}


def run_turn(graph, stream_input, config) -> None:
    labels: dict[tuple, str] = {}             # subgraph namespace -> task description
    seen: set[str] = set()                    # tool_call_ids already rendered this turn
    while True:
        pending = {}                          # interrupt id -> Interrupt (may be several per step)
        stream = graph.stream(stream_input, config=config, stream_mode=["updates", "custom"], subgraphs=True)
        for ns, mode, payload in stream:
            if mode == "custom":
                if "description" in payload:
                    labels[ns] = payload["description"]
                continue
            if "__interrupt__" in payload:
                pending.update({i.id: i for i in payload["__interrupt__"]})
                continue
            for update in payload.values():
                render_update(update, label=labels.get(ns), seen=seen)
        if not pending:
            return
        stream_input = Command(resume={i.id: prompt_approval(i.value) for i in pending.values()})


def main() -> int:
    graph = build_agent()
    print("axon · /auto to toggle auto-approve · /exit to exit")
    config = {"configurable": {"thread_id": "1", "auto_approve": False}}
    first_turn = True

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text == "/exit":
            return 0
        if text == "/auto":
            auto = not config["configurable"]["auto_approve"]
            config["configurable"]["auto_approve"] = auto
            print(f"auto-approve {'ON — tool calls run without asking' if auto else 'OFF'}")
            continue
        if not text:
            continue

        new_msgs: list[BaseMessage] = []
        if first_turn:
            new_msgs.append(SystemMessage(content=SYSTEM_PROMPT))
            first_turn = False
        new_msgs.append(HumanMessage(content=text))

        run_turn(graph, {"messages": new_msgs}, config)
        print()
