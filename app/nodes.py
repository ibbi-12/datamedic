"""Node implementations for the DataMedic agent graph."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app import prompts
from app.sandbox import execute_code

if TYPE_CHECKING:
    from app.graph import AgentState

MODEL_NAME = "llama-3.3-70b-versatile"
PROFILE_MAX_CHARS = 1500
MAX_QUALITY_REVIEWS = 1  # how many "ran fine but analytically weak" redos are allowed
EDA_QUESTION = "(auto-explore) What are the most interesting insights in this dataset?"

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=MODEL_NAME,
            temperature=0.2,
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    return _llm


def _ask(system_prompt: str, user_prompt: str) -> str:
    llm = _get_llm()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content.strip()


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _parse_json_reply(text: str) -> dict:
    """Parse a JSON object out of an LLM reply, tolerating code fences and stray prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


_METRICS_LINE_RE = re.compile(r"^METRICS_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


def extract_metrics(stdout: str) -> list[dict]:
    """Pull the structured METRICS_JSON line out of script stdout; [] if absent/broken."""
    match = _METRICS_LINE_RE.search(stdout)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    metrics = payload.get("metrics", [])
    if not isinstance(metrics, list):
        return []
    return [m for m in metrics if isinstance(m, dict) and m.get("label") and m.get("value")]


def profile_csv_text(csv_path: str) -> str:
    """Load a CSV and produce the profile string used in every prompt."""
    df = pd.read_csv(csv_path)
    lines = [
        f"shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "dtypes:",
        df.dtypes.to_string(),
        "",
        "null counts:",
        df.isnull().sum().to_string(),
        "",
        "sample rows:",
        df.head(5).to_string(),
    ]
    profile = "\n".join(lines)
    if len(profile) > PROFILE_MAX_CHARS:
        profile = profile[:PROFILE_MAX_CHARS] + "\n...[truncated]"
    return profile


def suggest_questions(csv_path: str) -> list[str]:
    """One-shot: propose 3 questions this CSV can answer. Used by the /suggest endpoint."""
    profile = profile_csv_text(csv_path)
    reply = _ask(
        prompts.SUGGEST_SYSTEM_PROMPT,
        prompts.SUGGEST_USER_TEMPLATE.format(csv_profile=profile),
    )
    questions = _parse_json_reply(reply).get("questions", [])
    return [q for q in questions if isinstance(q, str) and q.strip()][:3]


def profile_csv(state: AgentState) -> AgentState:
    """Deterministic: load the CSV and summarize its shape/dtypes/nulls/sample rows."""
    state["csv_profile"] = profile_csv_text(state["csv_path"])
    state["status"] = "planning"
    return state


def plan(state: AgentState) -> AgentState:
    """LLM: turn the profile + question into a short analysis plan.

    An empty question switches to auto-EDA mode: the model picks the most
    interesting analyses itself.
    """
    if state["question"].strip():
        state["mode"] = "question"
        user_prompt = prompts.PLAN_USER_TEMPLATE.format(
            csv_profile=state["csv_profile"], question=state["question"]
        )
        state["plan"] = _ask(prompts.PLAN_SYSTEM_PROMPT, user_prompt)
    else:
        state["mode"] = "eda"
        state["question"] = EDA_QUESTION
        user_prompt = prompts.PLAN_EDA_USER_TEMPLATE.format(csv_profile=state["csv_profile"])
        state["plan"] = _ask(prompts.PLAN_EDA_SYSTEM_PROMPT, user_prompt)

    state["status"] = "coding"
    return state


def write_code(state: AgentState) -> AgentState:
    """LLM: write (or rewrite, given prior failure/review) a standalone analysis script."""
    state["attempt"] = state.get("attempt", 0) + 1

    # The runner copies the CSV into the sandbox cwd as data.csv (see execute), so the
    # model never has to transcribe a long temp path — a reliable source of typo bugs.
    system_prompt = prompts.WRITE_CODE_SYSTEM_PROMPT.format(csv_path="data.csv")

    history = state.get("history", [])
    if history:
        last = history[-1]
        user_prompt = prompts.WRITE_CODE_RETRY_TEMPLATE.format(
            csv_profile=state["csv_profile"],
            question=state["question"],
            plan=state["plan"],
            attempt=state["attempt"] - 1,
            previous_code=last["code"],
            stderr=last["stderr"],
            critique=last["critique"],
        )
    else:
        user_prompt = prompts.WRITE_CODE_USER_TEMPLATE.format(
            csv_profile=state["csv_profile"], question=state["question"], plan=state["plan"]
        )

    raw = _ask(system_prompt, user_prompt)
    state["code"] = _strip_code_fence(raw)
    state["status"] = "executing"
    return state


def execute(state: AgentState) -> AgentState:
    """Deterministic: run the generated code in a fresh temp dir and capture the result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tempdir = Path(tmpdir)
        # place the data where the generated code expects it: ./data.csv in the sandbox
        (tempdir / "data.csv").write_bytes(Path(state["csv_path"]).read_bytes())
        result = execute_code(state["code"], tempdir, tempdir / "chart_1.png")

        state["stdout"] = result.stdout
        state["stderr"] = result.stderr

        if result.success:
            state["metrics"] = extract_metrics(result.stdout)
            # Persist charts next to the job's own CSV, since the subprocess temp dir
            # is deleted as soon as this block exits.
            job_dir = Path(state["csv_path"]).parent
            charts: list[str] = []
            for src in sorted(tempdir.glob("chart_*.png")):
                dest = job_dir / src.name
                dest.write_bytes(src.read_bytes())
                charts.append(str(dest))
            state["charts"] = charts
            state["chart_path"] = charts[0] if charts else None
            # Not "done" yet — review, summarize, and verify still have to run.
            state["status"] = "reviewing"
        else:
            state["charts"] = []
            state["chart_path"] = None
            state["status"] = "fixing"

    return state


def review(state: AgentState) -> AgentState:
    """LLM quality critic: the script ran, but does its output actually answer the question?

    Analytically weak output is sent back to write_code (within a small budget) with
    the reviewer's feedback taking the place of a crash critique.
    """
    user_prompt = prompts.REVIEW_USER_TEMPLATE.format(
        question=state["question"],
        plan=state["plan"],
        stdout=state["stdout"],
        n_charts=len(state.get("charts", [])),
    )
    try:
        verdict = _parse_json_reply(_ask(prompts.REVIEW_SYSTEM_PROMPT, user_prompt))
    except (ValueError, json.JSONDecodeError):
        verdict = {"verdict": "approve", "feedback": ""}  # unparseable review never blocks

    wants_revision = verdict.get("verdict") == "revise" and bool(verdict.get("feedback"))
    budget_left = (
        state.get("reviews_used", 0) < MAX_QUALITY_REVIEWS
        and state["attempt"] < state["max_attempts"]
    )

    if wants_revision and budget_left:
        state["reviews_used"] = state.get("reviews_used", 0) + 1
        history = state.get("history", [])
        history.append(
            {
                "code": state["code"],
                "stderr": "[quality review] the script ran, but the reviewer rejected the analysis",
                "critique": verdict["feedback"],
            }
        )
        state["history"] = history
        state["status"] = "fixing"
    else:
        state["status"] = "summarizing"

    return state


def critique(state: AgentState) -> AgentState:
    """LLM, only on failure: diagnose the bug and record the attempt in history."""
    user_prompt = prompts.CRITIQUE_USER_TEMPLATE.format(code=state["code"], stderr=state["stderr"])
    critique_text = _ask(prompts.CRITIQUE_SYSTEM_PROMPT, user_prompt)

    history = state.get("history", [])
    history.append({"code": state["code"], "stderr": state["stderr"], "critique": critique_text})
    state["history"] = history

    state["status"] = "fixing" if state["attempt"] < state["max_attempts"] else "failed"
    return state


def summarize(state: AgentState) -> AgentState:
    """LLM, on success: turn stdout into a plain-English written insight."""
    user_prompt = prompts.SUMMARIZE_USER_TEMPLATE.format(
        question=state["question"], plan=state["plan"], stdout=state["stdout"]
    )
    state["result_summary"] = _ask(prompts.SUMMARIZE_SYSTEM_PROMPT, user_prompt)
    state["status"] = "verifying"
    return state


def verify(state: AgentState) -> AgentState:
    """LLM fact-checker: every number in the summary must be grounded in the stdout."""
    user_prompt = prompts.VERIFY_USER_TEMPLATE.format(
        summary=state["result_summary"], stdout=state["stdout"]
    )
    try:
        verdict = _parse_json_reply(_ask(prompts.VERIFY_SYSTEM_PROMPT, user_prompt))
        if not verdict.get("accurate", True) and verdict.get("corrected_summary"):
            state["result_summary"] = verdict["corrected_summary"]
        state["verified"] = True
    except (ValueError, json.JSONDecodeError):
        state["verified"] = False  # verification failed to run; don't claim the badge

    state["status"] = "done"
    return state
