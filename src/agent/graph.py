from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send, interrupt

from tools.task_tool import task
from tools.todo_tools import Todo

TASK = task.__name__
STEP_LIMIT_PROMPT = (
    "Step limit reached. Do not call any more tools. Summarize what you found "
    "or changed so far and state clearly what is left unfinished."
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    todos: list[Todo]
    steps: int
    description: str


def pending_tool_calls(state: State) -> list[dict]:
    ai = next(m for m in reversed(state["messages"]) if getattr(m, "tool_calls", None))
    answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
    return [tc for tc in ai.tool_calls if tc["id"] not in answered]


def build_graph(
    llm: BaseChatModel,
    tools: list,
    hitl: bool,
    max_steps: int | None = None,
    subagents: dict | None = None,
    checkpointer=None,
):
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools([*tools, task] if subagents else tools)

    def chat_node(state: State) -> dict:
        steps = state.get("steps", 0) + 1
        if steps == 1 and state.get("description"):
            get_stream_writer()({"description": state["description"]})
        if max_steps is not None and steps > max_steps:
            nudge = HumanMessage(content=STEP_LIMIT_PROMPT)
            summary = llm.invoke([*state["messages"], nudge])
            return {"messages": [nudge, summary], "steps": steps}
        return {"messages": [llm_with_tools.invoke(state["messages"])], "steps": steps}

    def human_approval_node(state: State, config: RunnableConfig) -> dict:
        if config["configurable"].get("auto_approve"):      # /auto -> run without asking
            return {"messages": []}

        rejections = []
        for tc in state["messages"][-1].tool_calls:
            decision = interrupt({
                "name": tc["name"],
                "args": tc["args"],
                "subagent": state.get("description", ""),
            })
            if not decision["approved"]:
                rejections.append(ToolMessage(
                    content=f"User rejected this call. Reason: {decision['reason']}",
                    tool_call_id=tc["id"],
                ))
        return {"messages": rejections}

    def tool_node(state: State) -> dict:
        working = state                       # reflects todos written earlier in this same turn
        update = {}
        messages = []
        for tc in pending_tool_calls(state):
            if tc["name"] == TASK:            # dispatched to the subagent node via Send
                continue
            tool = tools_by_name[tc["name"]]
            result = tool.invoke({**tc["args"], "state": working})
            if tc["name"] == "write_todos":
                update["todos"] = tc["args"]["todos"]
                working = {**working, "todos": tc["args"]["todos"]}
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": messages, **update}

    def subagent_node(tc: dict, config: RunnableConfig) -> dict:
        args = tc["args"]
        try:
            if args["subagent_type"] not in subagents:
                raise ValueError(
                    f"Unknown subagent_type {args['subagent_type']!r}. "
                    f"Use one of: {', '.join(subagents)}."
                )
            sub = subagents[args["subagent_type"]]
            result = sub["graph"].invoke(
                {
                    "messages": [SystemMessage(content=sub["prompt"]), HumanMessage(content=args["prompt"])],
                    "description": args["description"],
                    "steps": 0,
                    "todos": [],
                },
                config,
            )
            content = result["messages"][-1].content
        except GraphBubbleUp:                 # interrupt() inside the subgraph -> let it propagate
            raise
        except Exception as e:
            content = f"Subagent failed: {e!r}"
        return {"messages": [ToolMessage(content=content, tool_call_id=tc["id"], name=TASK)]}

    def dispatch(state: State) -> list:
        pending = pending_tool_calls(state)
        targets: list = [Send("subagent", tc) for tc in pending if tc["name"] == TASK]
        if any(tc["name"] != TASK for tc in pending):
            targets.append("tools")
        return targets or ["chat"]          # everything rejected -> let the model replan

    def route_after_chat(state: State) -> list | str:
        if not state["messages"][-1].tool_calls:
            return END
        return "human_approval" if hitl else dispatch(state)

    dispatch_targets = ["tools", "chat"] + (["subagent"] if subagents else [])

    builder = StateGraph(State)
    builder.add_node("chat", chat_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "chat")
    builder.add_edge("tools", "chat")
    if subagents:
        builder.add_node("subagent", subagent_node)
        builder.add_edge("subagent", "chat")
    if hitl:
        builder.add_node("human_approval", human_approval_node)
        builder.add_conditional_edges("chat", route_after_chat, ["human_approval", END])
        builder.add_conditional_edges("human_approval", dispatch, dispatch_targets)
    else:
        builder.add_conditional_edges("chat", route_after_chat, [*dispatch_targets, END])
    return builder.compile(checkpointer=checkpointer)
