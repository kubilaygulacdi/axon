from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import STEP_LIMIT_PROMPT, build_graph


class ScriptedChat(BaseChatModel):
    respond: Callable[[list, bool], AIMessage]
    tools_bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools_bound": True})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages, self.tools_bound))])


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


@tool
def touch(path: str) -> str:
    """Pretend to write a file."""
    return f"touched:{path}"


def call(name: str, tc_id: str, **args) -> dict:
    return {"name": name, "args": args, "id": tc_id, "type": "tool_call"}


def one_tool_then_answer(name: str, answer: str, **args):
    def respond(messages, tools_bound):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=answer)
        return AIMessage(content="", tool_calls=[call(name, f"{name}-1", **args)])
    return respond


def make_subagents(**kwargs) -> dict:
    sub_llm = ScriptedChat(respond=one_tool_then_answer("echo", "explore report", text="hi"))
    return {
        "explore": {
            "graph": build_graph(sub_llm, [echo], hitl=False, **kwargs),
            "prompt": "explore system prompt",
        }
    }


def test_dispatch_splits_task_and_ordinary_calls():
    def parent_respond(messages, tools_bound):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            call("task", "t1", description="look", prompt="find stuff", subagent_type="explore"),
            call("task", "t2", description="look again", prompt="find more", subagent_type="explore"),
            call("echo", "e1", text="parent"),
        ])

    graph = build_graph(ScriptedChat(respond=parent_respond), [echo], hitl=False, subagents=make_subagents())
    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    tool_msgs = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}
    assert set(tool_msgs) == {"t1", "t2", "e1"}
    assert tool_msgs["t1"].content == "explore report"
    assert tool_msgs["t2"].content == "explore report"
    assert tool_msgs["e1"].content == "echo:parent"
    assert result["messages"][-1].content == "done"
    assert not any(isinstance(m, ToolMessage) and m.content.startswith("echo:hi") for m in result["messages"])


def test_unknown_subagent_type_returns_error_tool_message():
    def parent_respond(messages, tools_bound):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            call("task", "t1", description="x", prompt="y", subagent_type="nope"),
        ])

    graph = build_graph(ScriptedChat(respond=parent_respond), [echo], hitl=False, subagents=make_subagents())
    result = graph.invoke({"messages": [HumanMessage(content="go")]})
    (tm,) = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tm.content.startswith("Subagent failed:")
    assert "nope" in tm.content


def test_step_cap_forces_tool_free_summary():
    calls = {"bound": 0, "unbound": 0}

    def respond(messages, tools_bound):
        if not tools_bound:
            calls["unbound"] += 1
            return AIMessage(content="partial summary")
        calls["bound"] += 1
        return AIMessage(content="", tool_calls=[call("echo", f"e{calls['bound']}", text="again")])

    graph = build_graph(ScriptedChat(respond=respond), [echo], hitl=False, max_steps=3)
    result = graph.invoke({"messages": [HumanMessage(content="loop forever")]})

    assert calls == {"bound": 3, "unbound": 1}
    assert sum(isinstance(m, ToolMessage) for m in result["messages"]) == 3
    assert any(isinstance(m, HumanMessage) and m.content == STEP_LIMIT_PROMPT for m in result["messages"])
    assert result["messages"][-1].content == "partial summary"
    assert result["steps"] == 4


def test_parallel_subagent_interrupts_resume_by_id():
    def parent_respond(messages, tools_bound):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            call("task", "t1", description="edit a", prompt="touch a", subagent_type="general"),
            call("task", "t2", description="edit b", prompt="touch b", subagent_type="general"),
        ])

    def sub_respond(messages, tools_bound):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"did {messages[-1].content}")
        path = messages[-1].content.split()[-1]
        return AIMessage(content="", tool_calls=[call("touch", f"touch-{path}", path=path)])

    subagents = {
        "general": {
            "graph": build_graph(ScriptedChat(respond=sub_respond), [touch], hitl=True),
            "prompt": "general system prompt",
        }
    }
    graph = build_graph(
        ScriptedChat(respond=parent_respond), [echo], hitl=True, subagents=subagents, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "t", "auto_approve": False}}

    rounds = []
    stream_input = {"messages": [HumanMessage(content="go")]}
    while True:
        pending = {}
        for ns, mode, payload in graph.stream(stream_input, config=config, stream_mode=["updates", "custom"], subgraphs=True):
            if mode == "updates" and "__interrupt__" in payload:
                pending.update({i.id: i for i in payload["__interrupt__"]})
        if not pending:
            break
        rounds.append(sorted((i.value["name"], i.value["subagent"]) for i in pending.values()))
        stream_input = Command(resume={i.id: {"approved": True, "reason": ""} for i in pending.values()})

    # parent approvals are sequential interrupt() calls inside one node -> one per round;
    # the two subagents run in parallel nodes -> their interrupts land in the same round
    assert rounds[0] == [("task", "")]
    assert rounds[1] == [("task", "")]
    assert rounds[2] == [("touch", "edit a"), ("touch", "edit b")]
    assert len(rounds) == 3

    final = graph.get_state(config).values["messages"]
    reports = {m.tool_call_id: m.content for m in final if isinstance(m, ToolMessage)}
    assert reports == {"t1": "did touched:a", "t2": "did touched:b"}
    assert final[-1].content == "done"


def test_subagent_emits_description_on_custom_stream():
    def parent_respond(messages, tools_bound):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            call("task", "t1", description="look", prompt="find stuff", subagent_type="explore"),
        ])

    graph = build_graph(ScriptedChat(respond=parent_respond), [echo], hitl=False, subagents=make_subagents())
    custom = [
        (ns, payload)
        for ns, mode, payload in graph.stream(
            {"messages": [HumanMessage(content="go")]}, stream_mode=["updates", "custom"], subgraphs=True
        )
        if mode == "custom"
    ]
    assert len(custom) == 1
    ns, payload = custom[0]
    assert payload == {"description": "look"}
    assert len(ns) == 1 and ns[0].startswith("subagent:")
