from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.llm import get_llm
from tools.file_tools import edit_file, glob, grep, list_dir, read_file, write_file
from tools.research_tools import web_search

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("axon")

mlflow.langchain.autolog()

TOOLS = [list_dir, read_file, glob, grep, edit_file, write_file, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
LLM = get_llm().bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chat_node(state: State) -> dict:
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": "1",   # same id for the whole REPL run
            "mlflow.trace.user": "admin",
        }
    )
    return {"messages": [LLM.invoke(state["messages"])]}


def tool_node(state: State) -> dict:
    last = state["messages"][-1]
    messages = []
    for tc in last.tool_calls:
        tool = TOOLS_BY_NAME[tc["name"]]
        result = tool.invoke(tc["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": messages}


def route_after_chat(state: State) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


def build_graph():
    builder = StateGraph(State)
    builder.add_node("chat", chat_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "chat")
    builder.add_conditional_edges("chat", route_after_chat)
    builder.add_edge("tools", "chat")
    return builder.compile(checkpointer=InMemorySaver())
