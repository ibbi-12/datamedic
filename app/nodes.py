"""Node implementations for the DataMedic agent graph."""

from __future__ import annotations

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
CHART_FILENAME = "chart.png"

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


def profile_csv(state: AgentState) -> AgentState:
    """Deterministic: load the CSV and summarize its shape/dtypes/nulls/sample rows."""
    df = pd.read_csv(state["csv_path"])
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

    state["csv_profile"] = profile
    state["status"] = "planning"
    return state


def plan(state: AgentState) -> AgentState:
    """LLM: turn the profile + question into a short analysis plan."""
    user_prompt = prompts.PLAN_USER_TEMPLATE.format(
        csv_profile=state["csv_profile"], question=state["question"]
    )
    state["plan"] = _ask(prompts.PLAN_SYSTEM_PROMPT, user_prompt)
    state["status"] = "coding"
    return state


def write_code(state: AgentState) -> AgentState:
    """LLM: write (or rewrite, given prior failure) a standalone analysis script."""
    state["attempt"] = state.get("attempt", 0) + 1

    system_prompt = prompts.WRITE_CODE_SYSTEM_PROMPT.format(
        csv_path=state["csv_path"], chart_path=CHART_FILENAME
    )

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
        chart_path = tempdir / CHART_FILENAME

        result = execute_code(state["code"], tempdir, chart_path)

        state["stdout"] = result.stdout
        state["stderr"] = result.stderr

        if result.success:
            # Not "done" yet — summarize still has to run. Flipping to "done" here
            # would let a polling client stop early and read result_summary empty.
            state["status"] = "summarizing"
            if result.chart_created:
                # Persist next to the job's own CSV, since the subprocess temp dir is
                # deleted as soon as this block exits.
                persist_path = Path(state["csv_path"]).parent / CHART_FILENAME
                persist_path.write_bytes(chart_path.read_bytes())
                state["chart_path"] = str(persist_path)
            else:
                state["chart_path"] = None
        else:
            state["status"] = "fixing"
            state["chart_path"] = None

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
    state["status"] = "done"
    return state
