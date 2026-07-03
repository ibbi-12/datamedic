"""Offline tests for the full agent graph: LLM calls are scripted, execution is real."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import nodes
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


def make_state(csv_path: str, question: str = "What is the revenue trend?") -> dict:
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
        "chart_path": None,
        "charts": [],
        "metrics": [],
        "reviews_used": 0,
        "verified": False,
        "result_summary": "",
        "history": [],
        "status": "planning",
    }


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("month,revenue\n2024-01,100\n2024-02,150\n2024-03,90\n")
    return str(path)


@pytest.fixture
def scripted_llm(monkeypatch):
    """Replace nodes._ask with a queue of canned replies, consumed in call order."""
    replies: list[str] = []

    def fake_ask(system_prompt: str, user_prompt: str) -> str:
        assert replies, "graph made more LLM calls than the test scripted"
        return replies.pop(0)

    monkeypatch.setattr(nodes, "_ask", fake_ask)
    return replies


def test_crash_then_heal_then_verify(csv_file, scripted_llm):
    """Attempt 1 crashes, critique routes back, attempt 2 passes review and verification."""
    scripted_llm.extend(
        [
            "Aggregate revenue by month and plot a line chart.",  # plan
            f"```python\n{CRASHING_CODE}\n```",                   # write_code #1
            "KeyError: aggregate before indexing.",               # critique
            f"```python\n{GOOD_CODE}\n```",                       # write_code #2
            APPROVE,                                              # review
            "Revenue peaked in May 2024 at $19,586.",             # summarize
            ACCURATE,                                             # verify
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
    assert not scripted_llm, "unused scripted replies left over"


def test_reviewer_sends_weak_analysis_back(csv_file, scripted_llm):
    """The script runs fine but the reviewer rejects it; the rewrite is approved."""
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
    assert "aggregate by month" in final["history"][0]["critique"]


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
        ]
    )

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "done"
    assert final["history"][0]["stderr"] == "blocked import: os"


def test_exhausted_attempts_end_failed(csv_file, scripted_llm):
    """Four crashes in a row exhaust the budget and end with status failed."""
    scripted_llm.append("Plan.")
    for _ in range(4):
        scripted_llm.append(f"```python\n{CRASHING_CODE}\n```")  # write_code
        scripted_llm.append("Still broken.")                     # critique

    final = build_graph().invoke(make_state(csv_file))

    assert final["status"] == "failed"
    assert final["attempt"] == 4
    assert len(final["history"]) == 4
    assert final["result_summary"] == ""


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
