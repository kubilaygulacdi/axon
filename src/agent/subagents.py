from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from agent.config import get_config
from agent.graph import build_graph
from agent.llm import get_llm
from tools.file_tools import edit_file, glob, grep, list_dir, read_file, write_file
from tools.research_tools import web_search
from tools.todo_tools import read_todos, write_todos

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_STEPS = 20

READ_ONLY_TOOLS = [list_dir, read_file, glob, grep, web_search]
WRITE_TOOLS = [edit_file, write_file]
PLANNING_TOOLS = [write_todos, read_todos]


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def build_subagents() -> dict:
    cfg = get_config()
    return {
        "explore": {
            "graph": build_graph(get_llm(cfg.llm_explore_model), READ_ONLY_TOOLS, hitl=False, max_steps=MAX_STEPS),
            "prompt": load_prompt("explore"),
        },
        "general": {
            "graph": build_graph(get_llm(), READ_ONLY_TOOLS + WRITE_TOOLS, hitl=True, max_steps=MAX_STEPS),
            "prompt": load_prompt("general"),
        },
    }


def build_agent():
    return build_graph(
        get_llm(),
        READ_ONLY_TOOLS + WRITE_TOOLS + PLANNING_TOOLS,
        hitl=True,
        subagents=build_subagents(),
        checkpointer=InMemorySaver(),
    )
