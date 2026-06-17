from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from agent.graph import build_graph

SYSTEM_PROMPT = "You are Axon a terminal-native AI assistant."

TOOL_OUTPUT_LIMIT = 400


def render_update(update: dict) -> None:
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"[tool] {tc['name']}({args})")
            if msg.content and not msg.tool_calls:
                print("\n[agent]")
                print(msg.content)
        elif isinstance(msg, ToolMessage):
            content = msg.content
            full_len = len(content)
            if full_len > TOOL_OUTPUT_LIMIT:
                content = content[:TOOL_OUTPUT_LIMIT] + f"\n... [{full_len} chars total]"
            for line in content.splitlines() or [""]:
                print(f"   │ {line}")


def render_preview(name: str, args: dict) -> str:
    lines = [f"approve> {name}"]
    for key, value in args.items():
        text = value if isinstance(value, str) else repr(value)
        if len(text) > TOOL_OUTPUT_LIMIT:
            text = text[:TOOL_OUTPUT_LIMIT] + f"\n... [{len(text)} chars total]"
        body = text.splitlines() or [""]
        if len(body) == 1:
            lines.append(f"  {key}: {body[0]}")
        else:
            lines.append(f"  {key}:")
            lines.extend(f"  │ {line}" for line in body)
    return "\n".join(lines)


def prompt_approval(call: dict) -> dict:
    print(render_preview(call["name"], call["args"]))
    if input("approve? [y/N]: ").strip().lower() in ("y", "yes"):
        return {"approved": True, "reason": ""}
    reason = input("reason (sent back to the model): ").strip()
    return {"approved": False, "reason": reason or "No reason given."}


def run_turn(graph, stream_input, config) -> None:
    while True:
        interrupted = False
        for event in graph.stream(stream_input, config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                payload = event["__interrupt__"][0].value
                stream_input = Command(resume=prompt_approval(payload))
                interrupted = True
                break
            for update in event.values():
                render_update(update)
        if not interrupted:
            return


def main() -> int:
    graph = build_graph()
    print("axon · /exit to exit")
    config = {"configurable": {"thread_id": "1"}}
    first_turn = True

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text == "/exit":
            return 0
        if not text:
            continue

        new_msgs: list[BaseMessage] = []
        if first_turn:
            new_msgs.append(SystemMessage(content=SYSTEM_PROMPT))
            first_turn = False
        new_msgs.append(HumanMessage(content=text))

        run_turn(graph, {"messages": new_msgs}, config)
        print()
