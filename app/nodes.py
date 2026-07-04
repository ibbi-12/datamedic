"""Node implementations for the DataMedic agent graph."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import groq
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app import memory, prompts
from app.events import current_stream, emit
from app.sandbox import ExecutionResult, execute_code

if TYPE_CHECKING:
    from app.graph import AgentState

MODEL_NAME = os.environ.get("DATAMEDIC_MODEL", "llama-3.3-70b-versatile")
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
            # We own retrying in _ask (and want to skip it entirely for a hard
            # daily quota) — the SDK's own silent retries would just burn time
            # re-hitting the same 429 before we ever see it.
            max_retries=0,
        )
    return _llm


LLM_ATTEMPTS = 3
LLM_BACKOFF_SECONDS = 15  # short transient errors only; see _is_daily_quota_error


def _is_daily_quota_error(exc: Exception) -> bool:
    """True for Groq's 'tokens per day' cap — retrying won't help for minutes."""
    return isinstance(exc, groq.RateLimitError) and "per day" in str(exc).lower()


class DailyQuotaExceeded(RuntimeError):
    """Raised in place of the raw 429 so callers/UI get an actionable message."""


def _ask(system_prompt: str, user_prompt: str, node: str | None = None) -> str:
    """One LLM call with rate-limit resilience. When `node` is given, tokens are
    streamed to the job's event bus."""
    llm = _get_llm()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    last_exc: Exception | None = None
    for attempt in range(LLM_ATTEMPTS):
        if attempt:
            emit("llm_retry", node=node or "", attempt=attempt)
            time.sleep(LLM_BACKOFF_SECONDS * attempt)
        try:
            if node is None:
                return llm.invoke(messages).content.strip()

            parts: list[str] = []
            try:
                for chunk in llm.stream(messages):
                    piece = chunk.content or ""
                    if piece:
                        parts.append(piece)
                        emit("token", node=node, text=piece)
            except Exception:
                if not parts:  # stream never started; fall back to a plain call
                    return llm.invoke(messages).content.strip()
            emit("token_end", node=node)
            return "".join(parts).strip()
        except Exception as exc:  # rate limit / transient network
            if _is_daily_quota_error(exc):
                raise DailyQuotaExceeded(
                    f"Daily token quota reached for model '{MODEL_NAME}'. "
                    "Retrying won't help for several minutes. Set DATAMEDIC_MODEL to a "
                    "different model (e.g. llama-3.1-8b-instant, a separate quota "
                    "bucket) or wait for the daily reset. Groq said: "
                    f"{exc}"
                ) from exc
            last_exc = exc  # transient — worth a short backoff and retry
    raise last_exc  # type: ignore[misc]


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


def _ask_json(system_prompt: str, user_prompt: str, tries: int = 2) -> dict:
    """LLM call that must return JSON; re-asks once if the reply doesn't parse."""
    last_exc: Exception | None = None
    for _ in range(tries):
        try:
            return _parse_json_reply(_ask(system_prompt, user_prompt))
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


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
    payload = _ask_json(
        prompts.SUGGEST_SYSTEM_PROMPT,
        prompts.SUGGEST_USER_TEMPLATE.format(csv_profile=profile),
    )
    questions = payload.get("questions", [])
    return [q for q in questions if isinstance(q, str) and q.strip()][:3]


# --- sandbox plumbing shared by execute and race ---


def _run_sandboxed(code: str, csv_path: str) -> tuple[ExecutionResult, list[tuple[str, bytes]]]:
    """Run code in a fresh sandbox dir (CSV provided as ./data.csv); charts as blobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tempdir = Path(tmpdir)
        (tempdir / "data.csv").write_bytes(Path(csv_path).read_bytes())
        result = execute_code(code, tempdir, tempdir / "chart_1.png")
        chart_blobs = (
            [(p.name, p.read_bytes()) for p in sorted(tempdir.glob("chart_*.png"))]
            if result.success
            else []
        )
    return result, chart_blobs


def _apply_run(state: AgentState, result: ExecutionResult, chart_blobs: list) -> None:
    """Fold an execution result into the state, persisting charts on success."""
    state["stdout"] = result.stdout
    state["stderr"] = result.stderr
    if result.success:
        state["metrics"] = extract_metrics(result.stdout)
        job_dir = Path(state["csv_path"]).parent
        charts: list[str] = []
        for name, blob in chart_blobs:
            dest = job_dir / name
            dest.write_bytes(blob)
            charts.append(str(dest))
        state["charts"] = charts
        state["chart_path"] = charts[0] if charts else None
        # Not "done" yet — review, summarize, and verify still have to run.
        state["status"] = "reviewing"
    else:
        state["charts"] = []
        state["chart_path"] = None
        state["status"] = "fixing"


def _initial_code_prompts(state: AgentState) -> tuple[str, str]:
    """System + user prompt for a first attempt, with any learned lessons injected."""
    system_prompt = prompts.WRITE_CODE_SYSTEM_PROMPT.format(csv_path="data.csv")
    user_prompt = prompts.WRITE_CODE_USER_TEMPLATE.format(
        csv_profile=state["csv_profile"], question=state["question"], plan=state["plan"]
    )
    lessons = memory.retrieve_lessons(state["csv_profile"])
    if lessons:
        state["lessons_used"] = [l["symptom"] for l in lessons]
        user_prompt += prompts.LESSONS_BLOCK_TEMPLATE.format(
            lessons=memory.format_lessons(lessons)
        )
        emit("lessons_applied", lessons=state["lessons_used"])
    return system_prompt, user_prompt


# --- graph nodes ---


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
        state["plan"] = _ask(prompts.PLAN_SYSTEM_PROMPT, user_prompt, node="plan")
    else:
        state["mode"] = "eda"
        state["question"] = EDA_QUESTION
        user_prompt = prompts.PLAN_EDA_USER_TEMPLATE.format(csv_profile=state["csv_profile"])
        state["plan"] = _ask(prompts.PLAN_EDA_SYSTEM_PROMPT, user_prompt, node="plan")

    state["status"] = "coding"
    return state


def write_code(state: AgentState) -> AgentState:
    """LLM: write (or rewrite, given prior failure/review) a standalone analysis script."""
    state["attempt"] = state.get("attempt", 0) + 1

    history = state.get("history", [])
    if history:
        system_prompt = prompts.WRITE_CODE_SYSTEM_PROMPT.format(csv_path="data.csv")
        last = history[-1]
        earlier = history[:-1]
        if earlier:
            failures = "\n".join(
                f"- attempt {i + 1}: {h['stderr'].strip().splitlines()[-1][:160]}"
                f" (advice given: {h['critique'][:160]})"
                for i, h in enumerate(earlier)
            )
            prior_failures = prompts.PRIOR_FAILURES_TEMPLATE.format(failures=failures)
        else:
            prior_failures = ""
        user_prompt = prompts.WRITE_CODE_RETRY_TEMPLATE.format(
            csv_profile=state["csv_profile"],
            question=state["question"],
            plan=state["plan"],
            prior_failures=prior_failures,
            attempt=state["attempt"] - 1,
            previous_code=last["code"],
            stderr=last["stderr"],
            critique=last["critique"],
        )
    else:
        system_prompt, user_prompt = _initial_code_prompts(state)

    raw = _ask(system_prompt, user_prompt, node="write_code")
    state["code"] = _strip_code_fence(raw)
    state["status"] = "executing"
    return state


def execute(state: AgentState) -> AgentState:
    """Deterministic: run the generated code in a fresh sandbox and fold in the result."""
    result, chart_blobs = _run_sandboxed(state["code"], state["csv_path"])
    _apply_run(state, result, chart_blobs)
    return state


def race(state: AgentState) -> AgentState:
    """Attempt 1, race mode: rival coders with different strategies run in parallel.

    All candidates are generated and executed concurrently; a judge LLM picks the
    best successful one. If everything crashes, the first candidate's failure is
    handed to the normal critique path.
    """
    n = max(1, min(state.get("race_n", 1), len(prompts.RACE_STRATEGIES)))
    state["attempt"] = 1
    state["status"] = "executing"

    system_prompt, base_user_prompt = _initial_code_prompts(state)
    stream = current_stream()  # worker threads don't inherit the contextvar

    def s_emit(kind: str, **data) -> None:
        if stream is not None:
            stream.emit(kind, **data)

    s_emit("race_start", n=n, strategies=prompts.RACE_STRATEGIES[:n])

    def run_candidate(i: int) -> dict:
        s_emit("race_candidate", index=i, phase="writing")
        user_prompt = f"{base_user_prompt}\n\nStyle directive: {prompts.RACE_STRATEGIES[i]}"
        code = _strip_code_fence(_ask(system_prompt, user_prompt))
        s_emit("race_candidate", index=i, phase="running")
        result, blobs = _run_sandboxed(code, state["csv_path"])
        s_emit("race_candidate", index=i, phase="passed" if result.success else "crashed")
        return {"code": code, "result": result, "blobs": blobs}

    with ThreadPoolExecutor(max_workers=n) as pool:
        candidates = list(pool.map(run_candidate, range(n)))

    winners = [i for i, c in enumerate(candidates) if c["result"].success]
    reason = ""
    if len(winners) > 1:
        blocks = "\n\n".join(
            f"Candidate {i} stdout:\n{candidates[i]['result'].stdout[:800]}" for i in winners
        )
        try:
            verdict = _ask_json(
                prompts.RACE_JUDGE_SYSTEM_PROMPT,
                prompts.RACE_JUDGE_USER_TEMPLATE.format(
                    question=state["question"], plan=state["plan"], candidates=blocks
                ),
            )
            picked = int(verdict.get("winner", winners[0]))
            reason = verdict.get("reason", "")
            winner = picked if picked in winners else winners[0]
        except (ValueError, TypeError, json.JSONDecodeError):
            winner = winners[0]
    elif winners:
        winner = winners[0]
    else:
        winner = 0  # everything crashed; critique the first candidate

    state["race_report"] = {
        "strategies": prompts.RACE_STRATEGIES[:n],
        "outcomes": ["passed" if c["result"].success else "crashed" for c in candidates],
        "winner": winner if winners else None,
        "reason": reason,
    }
    s_emit("race_end", **state["race_report"])

    chosen = candidates[winner]
    state["code"] = chosen["code"]
    _apply_run(state, chosen["result"], chosen["blobs"])
    return state


def review(state: AgentState) -> AgentState:
    """LLM quality critic: the script ran, but does its output actually answer the question?

    Analytically weak output is sent back to write_code (within a small budget) with
    the reviewer's feedback taking the place of a crash critique.
    """
    user_prompt = prompts.REVIEW_USER_TEMPLATE.format(
        question=state["question"],
        plan=state["plan"],
        code=state["code"],
        stdout=state["stdout"],
        n_charts=len(state.get("charts", [])),
    )
    try:
        verdict = _ask_json(prompts.REVIEW_SYSTEM_PROMPT, user_prompt)
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
    critique_text = _ask(prompts.CRITIQUE_SYSTEM_PROMPT, user_prompt, node="critique")

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
    state["result_summary"] = _ask(prompts.SUMMARIZE_SYSTEM_PROMPT, user_prompt, node="summarize")
    state["status"] = "verifying"
    return state


def verify(state: AgentState) -> AgentState:
    """LLM fact-checker: every number in the summary must be grounded in the stdout."""
    user_prompt = prompts.VERIFY_USER_TEMPLATE.format(
        summary=state["result_summary"], stdout=state["stdout"]
    )
    try:
        verdict = _ask_json(prompts.VERIFY_SYSTEM_PROMPT, user_prompt)
        if not verdict.get("accurate", True) and verdict.get("corrected_summary"):
            state["result_summary"] = verdict["corrected_summary"]
        state["verified"] = True
    except (ValueError, json.JSONDecodeError):
        state["verified"] = False  # verification failed to run; don't claim the badge

    state["status"] = "done"
    return state


def learn(state: AgentState) -> AgentState:
    """LLM librarian: distill crashes into reusable lessons.

    Runs after healed runs AND after exhausted ones — a run that kept crashing is
    exactly the run worth learning from. Quality-review rejections aren't crashes
    and don't generalize the same way, so only genuine failures are distilled.
    Any error here is swallowed — learning must never break a finished run.
    """
    crashes = [
        h for h in state.get("history", [])
        if not h["stderr"].startswith("[quality review]")
    ]
    if not crashes:
        return state

    failures = "\n\n".join(
        f"stderr:\n{h['stderr'][:600]}\ndiagnosis: {h['critique']}" for h in crashes
    )
    final_code = (
        state["code"]
        if state["status"] != "failed"
        else "(no attempt ever succeeded — distill lessons from the failures alone)"
    )
    try:
        payload = _ask_json(
            prompts.DISTILL_SYSTEM_PROMPT,
            prompts.DISTILL_USER_TEMPLATE.format(
                csv_profile=state["csv_profile"],
                failures=failures,
                final_code=final_code,
            ),
        )
        lessons = payload.get("lessons", [])
        stored = memory.save_lessons(lessons)
        state["lessons_learned"] = stored
        if stored:
            emit("lessons_learned", count=stored, lessons=[l.get("symptom", "") for l in lessons])
    except Exception:
        logging.exception("librarian failed to distill lessons")  # never fail the run

    return state
