from typing import Literal

from pydantic import BaseModel, Field


class task(BaseModel):
    """Delegate a self-contained piece of work to a subagent with its own context window.

    The subagent starts fresh: it sees only the prompt you give it, works with
    its own tools, and returns a single text report. None of its intermediate
    tool output lands in your context -- that is the point. Use it when the
    work would otherwise flood your context with file dumps or search results
    you only need the conclusion of.

    When to use:
        - Broad codebase sweeps ("find every place we handle auth tokens and
          summarize the flow") -> subagent_type="explore".
        - A self-contained edit that touches several files and can be fully
          specified up front ("rename get_llm to build_llm across src/ and
          update callers") -> subagent_type="general".
        - Independent questions you want answered in parallel: emit several
          task calls in ONE message and they run concurrently.

    When NOT to use:
        - A single-fact lookup where you already know the file or symbol.
          Just call read_file or grep yourself; a subagent is slower.
        - Work that depends on your conversation history. The subagent cannot
          see it, so anything it needs must be in the prompt.
        - Follow-up questions to a finished subagent. There is no channel back;
          start a new task with a sharper prompt instead.

    The prompt must be complete on its own: what to find or change, where to
    look, what form the answer should take. Say explicitly what you want
    returned (file paths, line numbers, a diff summary, a yes/no with
    evidence). Subagents stop after 20 tool rounds and return whatever they
    have, so ask for a focused deliverable, not "look at everything".

    Example:
        task(
            description="map auth flow",
            prompt="Find where user login is handled under src/. Report the "
                   "entry function, the files it touches, and how the session "
                   "token is stored. Give file paths with line numbers.",
            subagent_type="explore",
        )
    """

    description: str = Field(
        description="Three to five word label shown to the user while the subagent runs, e.g. 'map auth flow'."
    )
    prompt: str = Field(
        description="The full, self-contained brief for the subagent. Include where to look and what exact form the answer should take."
    )
    subagent_type: Literal["explore", "general"] = Field(
        description="'explore' is read-only (list_dir, read_file, glob, grep, web_search) and runs on a cheaper model; use it for research and search. 'general' can also edit and write files; use it for delegated changes."
    )
