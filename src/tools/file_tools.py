import re
from pathlib import Path

from langchain_core.tools import tool

DEFAULT_READ_LIMIT = 2000  # row limit
MAX_LINE_LENGTH = 500  # char limit per row
GLOB_MAX_RESULTS = 200
GREP_MAX_RESULTS = 100
GREP_MAX_LINE_LENGTH = 200


@tool
def list_dir(path: str = ".") -> str:
    """List files and subdirectories in the given directory.

    Args:
        path: Directory path to list. Defaults to the current directory.

    Returns:
        Newline-separated entries; directories are suffixed with '/'.
    """
    target = Path(path).expanduser()
    if not target.exists(): 
        return (
            f"Error: '{path}' does not exist. "
            f"If unsure of the layout, run list_dir('.') first or "
            f"glob('**/{Path(path).name}*') to locate it."
        )
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)


@tool
def read_file(path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> str:
    """Read a text file with line numbers.
    Use this tool to read file contents
    For large files, use offset and limit to read specific sections.

    Args:
        path: File path to read.
        offset: 1-indexed line number to start reading from. Defaults to 1.
        limit: Maximum number of lines to return. Defaults to 2000.

    Returns:
        File content with line numbers prefixed (e.g. "    42\thello"),
        or an error message if the file cannot be read.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return (
            f"Error: '{path}' does not exist. "
            f"Use glob('**/{Path(path).name}') to find the correct path."
        )
    if not target.is_file():
        return f"Error: '{path}' is not a file."

    try:
        with target.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except PermissionError:
        return f"Error: permission denied for '{path}'."
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a UTF-8 text file (likely binary)."

    total = len(lines)
    if total == 0:
        return f"(empty file: {path})"

    start = max(0, offset - 1)
    if start >= total:
        return f"Error: offset {offset} exceeds file length ({total} lines)."

    selected = lines[start : start + limit]

    rendered = []
    for i, line in enumerate(selected, start=start + 1):
        line = line.rstrip("\n")
        if len(line) > MAX_LINE_LENGTH:
            extra = len(line) - MAX_LINE_LENGTH
            line = line[:MAX_LINE_LENGTH] + f"... [line truncated, {extra} more chars]"
        rendered.append(f"{i:6}\t{line}")

    output = "\n".join(rendered)
    end = start + len(selected)
    if end < total:
        output += f"\n\n[showing lines {start + 1}-{end} of {total}; use offset={end + 1} to continue]"
    return output


@tool
def glob(pattern: str, path: str = ".") -> str:
    """Find files and directories whose paths match a glob pattern.

    Use when you know roughly where a file lives or want every file of a
    certain type. For searching file *contents*, use `grep` instead. For
    listing a single directory's immediate children, `list_dir` is simpler.

    Pattern examples:
      '**/*.py'              — all Python files, recursive
      'src/*.toml'           — TOML files directly under src/
      'tests/**/test_*.py'   — test files in any tests subdirectory

    Returns newline-separated paths (directories suffixed with '/'), sorted.
    Capped at 200 matches; refine the pattern if truncated.

    Args:
        pattern: Glob pattern. Use '**' for recursive directory traversal.
        path: Base directory to search from. Defaults to the current directory.
    """
    base = Path(path).expanduser()
    if not base.exists():
        return f"Error: '{path}' does not exist."
    if not base.is_dir():
        return f"Error: '{path}' is not a directory."

    matches = sorted(base.glob(pattern), key=lambda p: str(p).lower())
    if not matches:
        return f"No matches for '{pattern}' in '{path}'."

    total = len(matches)
    truncated = total > GLOB_MAX_RESULTS
    if truncated:
        matches = matches[:GLOB_MAX_RESULTS]

    rendered = [f"{p}/" if p.is_dir() else str(p) for p in matches]
    output = "\n".join(rendered)
    if truncated:
        output += f"\n\n[showing {GLOB_MAX_RESULTS} of {total} matches; refine pattern to narrow]"
    return output


@tool
def grep(pattern: str, path: str = ".", include: str = "**/*") -> str:
    """Search for a regex pattern across the contents of files.

    Use when you need to find *where in the code* a symbol, string, or pattern
    appears — function definitions, references, TODOs, error messages. For
    finding files by *name* or *type*, use `glob`. For reading a single known
    file, use `read_file`.

    Pattern examples:
      'def \\w+_node'       — function definitions ending in _node
      'TODO|FIXME'          — common code markers
      'from agent\\.'       — imports of the agent package

    Filter the file set with `include` (a glob), e.g. `include='**/*.py'` to
    search only Python files. Binary and non-UTF-8 files are skipped silently.

    Returns matches as `path:lineno: line` (ripgrep style), sorted by path.
    Capped at 100 matches and 200 chars per line; tighten the pattern or
    `include` glob if truncated.

    Args:
        pattern: Python regular expression. Invalid regex returns an error string.
        path: Base directory to search in. Defaults to the current directory.
        include: Glob filter for which files to search. Defaults to all files.
    """
    base = Path(path).expanduser()
    if not base.exists():
        return f"Error: '{path}' does not exist."
    if not base.is_dir():
        return f"Error: '{path}' is not a directory."
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex '{pattern}': {e}"

    results = []
    truncated = False
    for file in sorted(base.glob(include), key=lambda p: str(p).lower()):
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                if len(line) > GREP_MAX_LINE_LENGTH:
                    line = line[:GREP_MAX_LINE_LENGTH] + f"... [+{len(line) - GREP_MAX_LINE_LENGTH} chars]"
                results.append(f"{file}:{lineno}: {line}")
                if len(results) >= GREP_MAX_RESULTS:
                    truncated = True
                    break
        if truncated:
            break

    if not results:
        return f"No matches for '{pattern}' in '{path}' (include='{include}')."

    output = "\n".join(results)
    if truncated:
        output += f"\n\n[truncated at {GREP_MAX_RESULTS} matches; tighten pattern or include glob]"
    return output


@tool
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace exact text in a file. Reads the current contents and rewrites
    the file with `old_string` substituted by `new_string`.

    Use this for surgical edits — fixing a bug, renaming a symbol, updating a
    config value. Always read the file first (`read_file`) so you know the
    exact text to match, including indentation and whitespace.

    Critical rules:
      - `old_string` must appear EXACTLY as written, including whitespace.
      - If `old_string` appears more than once and `replace_all=False`, the
        edit fails — add surrounding context to make the match unique, OR
        set `replace_all=True` to substitute every occurrence (good for
        renaming a variable across the file).
      - `old_string` and `new_string` must differ.
      - The file must already exist; this tool does not create new files.

    On success returns a confirmation with the replacement count. On any
    failure returns an error string explaining what went wrong, so you can
    retry with corrected arguments.

    Args:
        path: File to edit.
        old_string: Exact text to find. Must be unique unless replace_all=True.
        new_string: Replacement text.
        replace_all: If True, replace every occurrence. Defaults to False.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if not target.is_file():
        return f"Error: '{path}' is not a file."
    if old_string == new_string:
        return "Error: old_string and new_string are identical; nothing to change."

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a UTF-8 text file (likely binary)."
    except PermissionError:
        return f"Error: permission denied reading '{path}'."

    count = text.count(old_string)
    if count == 0:
        return (
            f"Error: old_string not found in '{path}'. "
            "Read the file again to verify exact whitespace and indentation."
        )
    if count > 1 and not replace_all:
        return (
            f"Error: old_string appears {count} times in '{path}'. "
            "Add more surrounding context to make the match unique, "
            "or set replace_all=True to replace every occurrence."
        )

    new_text = text.replace(old_string, new_string)
    try:
        target.write_text(new_text, encoding="utf-8")
    except PermissionError:
        return f"Error: permission denied writing '{path}'."

    suffix = "s" if count > 1 else ""
    return f"Edited '{path}' ({count} replacement{suffix})."


@tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a new file with the given content, or overwrite an existing one.

    Use when you need to create a file from scratch — a new module, config,
    test, or scratch note. For changing *part* of an existing file, prefer
    `edit_file` (surgical replacement) — it is far safer than rewriting the
    entire file.

    Safety rules:
      - By default this fails if the file already exists, to prevent
        accidental overwrites. Set `overwrite=True` to deliberately replace
        existing content (you should usually have read the file first).
      - Missing parent directories are created automatically.
      - Content is always written as UTF-8.

    On success returns a confirmation with the byte count. On failure returns
    an error string.

    Args:
        path: File path to write.
        content: Full file contents.
        overwrite: If True, replace an existing file. Defaults to False.
    """
    target = Path(path).expanduser()
    existed = target.exists()
    if existed:
        if target.is_dir():
            return f"Error: '{path}' is an existing directory."
        if not overwrite:
            return (
                f"Error: '{path}' already exists. "
                "Set overwrite=True to replace it, or use edit_file for surgical changes."
            )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return f"Error: permission denied creating directories for '{path}'."

    try:
        target.write_text(content, encoding="utf-8")
    except PermissionError:
        return f"Error: permission denied writing '{path}'."

    action = "Overwrote" if existed else "Wrote"
    return f"{action} '{path}' ({len(content.encode('utf-8'))} bytes)."
