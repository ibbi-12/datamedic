"""Offline tests for the full agent graph: LLM calls are scripted, execution is real."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import memory, nodes
from app.events import JobStream, bind, emit
from app.graph import build_graph
from app.nodes import _parse_json_reply, _strip_code_fence, extract_metrics

GOOD_CODE = """\
print("Revenue peaked in May 2024 at $19,586")
with open("chart_1.png", "wb") as f:
    f.write(b"fake png bytes")
with open("chart_2.png", "wb") as f:
    f.write(b"fake png bytes")
print('METRICS_JSON: {"metrics": [{"label": "Peak month", "value": "May 2024", "detail": "highest monthly revenue"}, {"label": "Peak revenue", "value": "$19,586"}]}')
"""

CRASHING_CODE = 'raise KeyError("revenue")'
BLOCKED_CODE = "import os\nprint(os.getcwd())"

APPROVE = '{"verdict": "approve", "feedback": ""}'
REVISE = '{"verdict": "revise", "feedback": "The output dumps raw rows; aggregate by month instead."}'
ACCURATE = '{"accurate": true, "corrected_summary": ""}'
CORRECTED = '{"accurate": false, "corrected_summary": "Corrected: revenue peaked in May 2024 at $19,586."}'
DISTILLED = json.dumps(
    {
        "lessons": [
            {
                "symptom": "Currency values stored as text",
                "lesson": "Strip '$' and ',' and cast to float before any arithmetic.",
                "triggers": ["$", "object"],
            }
        ]
    }
)
NO_LESSONS = '{"lessons": []}'
JUDGE_PICKS_1 = '{"winner": 1, "reason": "Candidate 1 answers more directly."}'


def make_state(csv_path: str, question: str = "What is the revenue trend?", race_n: int = 1) -> dict:
    return {
        "csv_path": csv_path,
        "csv_profile": "",
        "question": question,
        "mode": "question",
        "plan": "",
        "code": "",
        "stdout": "",
        "stderr": "",
        "attempt": 0,
        "max_attempts": 4,
        "race_n": race_n,
        "race_report": {},
        "chart_path": None,
        "charts": [],
        "metrics": [],
        "lessons_used": [],
        "lessons_learned": 0,
        "reviews_used": 0,
        "verified": False,
        "result_summary": "",
        "history": [],
        "status": "planning",
    }


@pytest.fixture(autouse=True)
def isolated_lessons(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMEDIC_LESSONS", str(tmp_path / "lessons.json"))


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("month,revenue\n2024-01,100\n2024-02,150\n2024-03,90\n")
    return str(path)


@pytest.fixture
def scripted_llm(monkeypatch):
    """Replace nodes._ask with a queue of canned replies, consumed in call order."""
    replies: list[str] = []
    prompts_seen: list[str] = []

    def fake_ask(system_prompt: str, user_prompt: str, node: str | None = None) -> str:
        assert replies, "graph made more LLM calls than the test scripted"
        prompts_seen.append(user_prompt)
        return replies.pop(0)

    monkeypatch.setattr(nodes, "_ask", fake_ask)
    replies_obj = replies
    replies_obj_prompts = prompts_seen
    fake_ask.prompts = prompts_seen  # type: ignore[attr-defined]
    return replies


def test_crash_then_heal_then_verify_then_learn(csv_file, scripted_llm):
    """Attempt 1 crashes, critique routes back, attempt 2 passes review, verification
    runs, and the librarian distills the crash into a stored lesson."""
    scripted_llm.extend(
        [
            "Aggregate revenue by month and plot a line chart.",  # plan
            f"```python\n{CRASHING_CODE}\n```",                   # write_code #1
            "KeyError: aggregate before indexing.",               # critique
            f"```python\n{GOOD_CODE}\n```",                       # write_code #2
            APPROVE,                                              # review
            "Revenue peaked in May 2024 at $19,586.",             # summarize
            ACCURATE,                                             # verify
            DISTILLED,                                            # learn
        ]
    )

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    assert final["attempt"] == 2
    assert len(final["history"]) == 1
    assert "KeyError" in final["history"][0]["stderr"]
    assert final["verified"] is True
    assert final["result_summary"].startswith("Revenue peaked")
    assert len(final["charts"]) == 2
    assert all(Path(c).exists() for c in final["charts"])
    assert final["metrics"][0]["label"] == "Peak month"
    assert final["lessons_learned"] == 1
    assert len(memory.load_lessons()) == 1
    assert not scripted_llm, "unused scripted replies left over"


def test_reviewer_sends_weak_analysis_back(csv_file, scripted_llm):
    """The script runs fine but the reviewer rejects it; the rewrite is approved.
    Quality rejections are not crashes, so the librarian is not consulted."""
    scripted_llm.extend(
        [
            "Aggregate revenue by month.",       # plan
            f"```python\n{GOOD_CODE}\n```",      # write_code #1 (runs, but reviewer rejects)
            REVISE,                              # review -> revise
            f"```python\n{GOOD_CODE}\n```",      # write_code #2
            APPROVE,                             # review -> approve
            "Revenue peaked in May 2024.",       # summarize
            ACCURATE,                            # verify
        ]
    )

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    assert final["attempt"] == 2
    assert final["reviews_used"] == 1
    assert len(final["history"]) == 1
    assert final["history"][0]["stderr"].startswith("[quality review]")
    assert memory.load_lessons() == []


def test_blocked_import_is_critiqued_and_recovered(csv_file, scripted_llm):
    """A blocked import short-circuits execution and routes through critique."""
    scripted_llm.extend(
        [
            "Plan.",                             # plan
            f"```python\n{BLOCKED_CODE}\n```",   # write_code #1 -> blocked import
            "Do not import os; hardcode the CSV path.",  # critique
            f"```python\n{GOOD_CODE}\n```",      # write_code #2
            APPROVE,                             # review
            "Summary.",                          # summarize
            ACCURATE,                            # verify
            NO_LESSONS,                          # learn (nothing generalizes)
        ]
    )

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    assert final["history"][0]["stderr"] == "blocked import: os"
    assert final["lessons_learned"] == 0


def test_exhausted_attempts_end_failed_but_still_learn(csv_file, scripted_llm):
    """Four crashes exhaust the budget; the run fails but the librarian still
    distills the failures into lessons for next time."""
    scripted_llm.append("Plan.")
    for _ in range(4):
        scripted_llm.append(f"```python\n{CRASHING_CODE}\n```")  # write_code
        scripted_llm.append("Still broken.")                     # critique
    scripted_llm.append(DISTILLED)                               # learn (on failure)

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "failed"
    assert final["attempt"] == 4
    assert len(final["history"]) == 4
    assert final["result_summary"] == ""
    assert final["lessons_learned"] == 1
    assert len(memory.load_lessons()) == 1


def test_verifier_replaces_hallucinated_summary(csv_file, scripted_llm):
    """A summary with numbers not in stdout gets replaced by the corrected one."""
    scripted_llm.extend(
        [
            "Plan.",
            f"```python\n{GOOD_CODE}\n```",
            APPROVE,
            "Revenue peaked in June 2025 at $99,999.",  # hallucinated summary
            CORRECTED,                                  # verify -> corrected
        ]
    )

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    assert final["result_summary"].startswith("Corrected:")
    assert final["verified"] is True


def test_eda_mode_on_empty_question(csv_file, scripted_llm):
    """An empty question switches to auto-EDA mode with its own plan prompt."""
    scripted_llm.extend(
        [
            "Explore monthly trend and biggest month-over-month change.",  # EDA plan
            f"```python\n{GOOD_CODE}\n```",
            APPROVE,
            "Summary.",
            ACCURATE,
        ]
    )

    final = build_graph().invoke(make_state(csv_file, question=""))

    assert final["status"] == "done"
    assert final["mode"] == "eda"
    assert final["question"] == nodes.EDA_QUESTION


def test_race_two_coders_judge_picks_winner(csv_file, scripted_llm):
    """race_n=2 runs rival coders in parallel; the judge picks among the successes."""
    scripted_llm.extend(
        [
            "Plan.",                          # plan
            f"```python\n{GOOD_CODE}\n```",   # candidate A (order with B nondeterministic)
            f"```python\n{GOOD_CODE}\n```",   # candidate B
            JUDGE_PICKS_1,                    # judge
            APPROVE,                          # review
            "Summary.",                       # summarize
            ACCURATE,                         # verify
        ]
    )

    final = build_graph().invoke(make_state(csv_file, race_n=2))

    assert final["status"] == "done"
    assert final["attempt"] == 1
    assert final["race_report"]["outcomes"] == ["passed", "passed"]
    assert final["race_report"]["winner"] == 1
    assert final["race_report"]["reason"]
    assert len(final["charts"]) == 2


def test_race_all_crash_routes_to_critique(csv_file, scripted_llm):
    """If every rival crashes, the failure flows into the normal critique/heal loop."""
    scripted_llm.extend(
        [
            "Plan.",                             # plan
            f"```python\n{CRASHING_CODE}\n```",  # candidate A
            f"```python\n{CRASHING_CODE}\n```",  # candidate B
            "KeyError diagnosis.",               # critique (no judge call: no successes)
            f"```python\n{GOOD_CODE}\n```",      # write_code retry (single, not raced)
            APPROVE,                             # review
            "Summary.",                          # summarize
            ACCURATE,                            # verify
            NO_LESSONS,                          # learn
        ]
    )

    final = build_graph().invoke(make_state(csv_file, race_n=2))

    assert final["status"] == "done"
    assert final["attempt"] == 2
    assert final["race_report"]["winner"] is None
    assert final["race_report"]["outcomes"] == ["crashed", "crashed"]


def test_learned_lessons_are_injected_into_coder_prompt(csv_file, scripted_llm, monkeypatch):
    """A stored lesson whose triggers match the profile lands in the coder's prompt."""
    memory.save_lessons(
        [
            {
                "symptom": "Revenue columns arrive as text",
                "lesson": "Cast revenue-like columns to numeric before aggregating.",
                "triggers": ["revenue"],
            }
        ]
    )

    seen_prompts: list[str] = []
    real_replies = [
        "Plan.",
        f"```python\n{GOOD_CODE}\n```",
        APPROVE,
        "Summary.",
        ACCURATE,
    ]

    def fake_ask(system_prompt: str, user_prompt: str, node: str | None = None) -> str:
        seen_prompts.append(user_prompt)
        return real_replies.pop(0)

    monkeypatch.setattr(nodes, "_ask", fake_ask)

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    assert final["lessons_used"] == ["Revenue columns arrive as text"]
    coder_prompt = seen_prompts[1]
    assert "Lessons learned from previous analyses" in coder_prompt
    assert "Cast revenue-like columns to numeric" in coder_prompt


def test_retry_prompt_carries_all_prior_failures(csv_file, monkeypatch):
    """By attempt 3 the coder must see BOTH earlier failures, not just the last one,
    so it stops cycling between two wrong fixes."""
    seen_prompts: list[str] = []
    replies = [
        "Plan.",
        f"```python\nraise KeyError('first-bug')\n```",   # attempt 1
        "First diagnosis.",                                # critique 1
        f"```python\nraise ValueError('second-bug')\n```", # attempt 2
        "Second diagnosis.",                               # critique 2
        f"```python\n{GOOD_CODE}\n```",                    # attempt 3
        APPROVE,
        "Summary.",
        ACCURATE,
        NO_LESSONS,
    ]

    def fake_ask(system_prompt: str, user_prompt: str, node: str | None = None) -> str:
        seen_prompts.append(user_prompt)
        return replies.pop(0)

    monkeypatch.setattr(nodes, "_ask", fake_ask)

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    third_coder_prompt = seen_prompts[5]
    assert "Earlier failed attempts" in third_coder_prompt
    assert "first-bug" in third_coder_prompt          # attempt 1's error is still visible
    assert "First diagnosis." in third_coder_prompt   # and the advice that failed
    assert "second-bug" in third_coder_prompt         # latest failure present in detail
    # attempt 2's prompt has only one prior failure -> no earlier-failures block
    second_coder_prompt = seen_prompts[3]
    assert "Earlier failed attempts" not in second_coder_prompt


# --- unit tests for memory ---


def test_memory_save_retrieve_and_dedupe():
    lesson = {"symptom": "Mixed date formats", "lesson": "Parse with format='mixed'.", "triggers": ["date", "mixed"]}
    assert memory.save_lessons([lesson]) == 1
    assert memory.save_lessons([lesson]) == 0  # dedupe by symptom

    hits = memory.retrieve_lessons("dtypes:\ndate  object\nsample rows: 05-Jan-2024")
    assert len(hits) == 1
    assert hits[0]["symptom"] == "Mixed date formats"

    # no trigger match -> recency fallback still surfaces recent lessons
    fallback = memory.retrieve_lessons("nothing relevant here")
    assert [l["symptom"] for l in fallback] == ["Mixed date formats"]
    assert "Parse with format" in memory.format_lessons(hits)


def test_memory_ignores_malformed_lessons():
    assert memory.save_lessons([{"symptom": "", "lesson": "x"}, "junk", {}]) == 0


# --- unit tests for events ---


def test_emit_is_noop_without_bound_stream():
    emit("token", node="plan", text="hello")  # must not raise


def test_bound_stream_collects_events():
    stream = JobStream()
    bind(stream)
    try:
        emit("node_start", node="plan", attempt=0)
        emit("token", node="plan", text="hi")
    finally:
        bind(None)
    kinds = [e["kind"] for e in stream.events]
    assert kinds == ["node_start", "token"]
    assert stream.events[1]["text"] == "hi"


# --- unit tests for the parsing helpers ---


def test_extract_metrics_happy_path():
    stdout = 'findings...\nMETRICS_JSON: {"metrics": [{"label": "Peak", "value": "$5"}]}\n'
    assert extract_metrics(stdout) == [{"label": "Peak", "value": "$5"}]


def test_extract_metrics_absent_or_broken():
    assert extract_metrics("no metrics here") == []
    assert extract_metrics("METRICS_JSON: {not json}") == []
    assert extract_metrics('METRICS_JSON: {"metrics": "not a list"}') == []


def test_parse_json_reply_tolerates_fences_and_prose():
    assert _parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_reply('Sure! {"verdict": "approve"} hope that helps') == {
        "verdict": "approve"
    }
    with pytest.raises(ValueError):
        _parse_json_reply("no json at all")


def test_strip_code_fence_variants():
    assert _strip_code_fence("```python\nx = 1\n```") == "x = 1"
    assert _strip_code_fence("x = 1") == "x = 1"


# --- unit tests for daily-quota fast-fail ---


def _fake_rate_limit_error(message: str):
    import groq
    import httpx

    req = httpx.Request("POST", "https://api.groq.com/x")
    resp = httpx.Response(429, request=req)
    return groq.RateLimitError(message, response=resp, body=None)


def test_daily_quota_error_is_detected():
    daily = _fake_rate_limit_error(
        "Error code: 429 - rate limit reached ... on tokens per day (TPD): Limit 100000"
    )
    per_minute = _fake_rate_limit_error("Error code: 429 - rate limit reached ... per minute")
    assert nodes._is_daily_quota_error(daily) is True
    assert nodes._is_daily_quota_error(per_minute) is False


def test_ask_fails_fast_on_daily_quota_without_retry_sleep(monkeypatch):
    """A TPD 429 must raise immediately with an actionable message, not burn
    LLM_ATTEMPTS worth of sleeps retrying a quota that won't clear for minutes."""
    calls = {"n": 0}

    class FakeLLM:
        def invoke(self, messages):
            calls["n"] += 1
            raise _fake_rate_limit_error(
                "Error code: 429 - ... on tokens per day (TPD): Limit 100000, Used 99950"
            )

    monkeypatch.setattr(nodes, "_get_llm", lambda: FakeLLM())
    slept = []
    monkeypatch.setattr(nodes.time, "sleep", lambda s: slept.append(s))

    with pytest.raises(nodes.DailyQuotaExceeded) as exc_info:
        nodes._ask("system", "user")

    assert calls["n"] == 1  # no retries burned on a hard daily cap
    assert slept == []
    assert "DATAMEDIC_MODEL" in str(exc_info.value)


def test_ask_still_retries_transient_errors(monkeypatch):
    """A non-quota error (e.g. a transient network blip) should still retry."""
    calls = {"n": 0}

    class FakeLLM:
        def invoke(self, messages):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient blip")
            return type("R", (), {"content": "ok"})()

    monkeypatch.setattr(nodes, "_get_llm", lambda: FakeLLM())
    monkeypatch.setattr(nodes.time, "sleep", lambda s: None)

    assert nodes._ask("system", "user") == "ok"
    assert calls["n"] == 2
