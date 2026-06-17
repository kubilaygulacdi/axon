from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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
        

        for event in graph.stream({"messages": new_msgs}, config=config, stream_mode="updates"):
            for update in event.values():
                render_update(update)
        print()
