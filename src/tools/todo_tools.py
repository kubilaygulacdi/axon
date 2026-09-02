from typing import Annotated, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

STATUS_ICON = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]


def render_todos(todos: list[Todo]) -> str:
    return "\n".join(
        f"{STATUS_ICON.get(t['status'], '[?]')} {t['content']}" for t in todos
    )


@tool
def write_todos(todos: list[Todo]) -> str:
    """Create or replace the task list used to plan multi-step work.

    Use this at the start of any non-trivial request to break it into concrete
    steps, then call it again after every step to update progress. The list is
    your working memory for long tasks -- rewriting it keeps you from drifting
    off the original goal over a long session.

    Do NOT use this for single, trivial actions (reading one file, answering a
    question). The overhead is not worth it.

    Rules:
        - Send the FULL list every time; this replaces the previous list.
          Revise, merge, or drop steps freely as you learn more.
        - Keep exactly one step 'in_progress' at a time.
        - Mark a step 'completed' the moment it is done, not in batches.
        - If a step is blocked, leave it 'in_progress' and add a new step
          describing what unblocks it.

    Args:
        todos: The complete task list. Each item needs 'content' (a short,
            actionable description) and 'status' (one of 'pending',
            'in_progress', 'completed').

    Returns:
        The rendered task list as it was saved.

    Example:
        write_todos([
            {"content": "Read graph.py to find the tool loop", "status": "in_progress"},
            {"content": "Add the todos field to State", "status": "pending"},
            {"content": "Run ruff check src", "status": "pending"},
        ])
    """
    if not todos:
        return "Task list cleared."
    return "Task list updated:\n" + render_todos(todos)


@tool
def read_todos(state: Annotated[dict, InjectedState]) -> str:
    """Re-read the current task list to check what is done and what is left.

    Call this after finishing a step, before deciding what to do next. On long
    tasks the plan scrolls far back in the conversation; this brings it back
    into view so you stay on the original goal.

    Returns:
        The task list with a status marker per step, or a note that no list
        exists yet (in which case call write_todos first).
    """
    todos = state.get("todos") or []
    if not todos:
        return "No task list yet. Call write_todos to create one."
    return render_todos(todos)
