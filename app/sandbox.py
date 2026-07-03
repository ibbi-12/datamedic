"""Static safety checks and subprocess execution for LLM-generated analysis code."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_SECONDS = 30

# (regex, human-readable name reported in the synthetic "blocked import: X" message)
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*import\s+os\b", "os"),
    (r"^\s*from\s+os\b", "os"),
    (r"^\s*import\s+sys\b", "sys"),
    (r"^\s*from\s+sys\b", "sys"),
    (r"\bsubprocess\b", "subprocess"),
    (r"\brequests\b", "requests"),
    (r"\burllib\b", "urllib"),
    (r"\bsocket\b", "socket"),
    (r"\bshutil\b", "shutil"),
    (r"\beval\s*\(", "eval("),
    (r"\bexec\s*\(", "exec("),
    (r"__import__", "__import__"),
]

# open("path", "w"/"a"/"x"...) where the literal path looks like it escapes the sandbox
_OPEN_CALL = re.compile(
    r"open\s*\(\s*(['\"])(?P<path>.*?)\1\s*(?:,\s*(['\"])(?P<mode>[rwaxb+t]+)\3)?"
)


def static_check(code: str) -> str | None:
    """Return a violation description if `code` trips a safety rule, else None."""
    for pattern, name in _BLOCKED_PATTERNS:
        if re.search(pattern, code, re.MULTILINE):
            return name

    for match in _OPEN_CALL.finditer(code):
        mode = match.group("mode") or "r"
        path = match.group("path")
        if any(c in mode for c in "wax") and (
            path.startswith("/") or path.startswith("~") or ".." in path
        ):
            return f"open() writing outside sandbox ({path!r})"

    return None


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    success: bool
    chart_created: bool


def execute_code(
    code: str,
    tempdir: Path,
    chart_path: Path,
    timeout: int = TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Statically check then run `code` as a subprocess inside `tempdir`."""
    violation = static_check(code)
    if violation:
        return ExecutionResult(
            stdout="",
            stderr=f"blocked import: {violation}",
            success=False,
            chart_created=False,
        )

    script_path = tempdir / "analysis.py"
    script_path.write_text(code)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(tempdir),
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += f"\n[TIMEOUT] execution exceeded {timeout}s"
        returncode = 1

    chart_created = chart_path.exists()
    success = returncode == 0 and bool(stdout.strip())

    return ExecutionResult(
        stdout=stdout, stderr=stderr, success=success, chart_created=chart_created
    )
