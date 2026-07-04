"""LangGraph build: state schema, node wiring, and edges."""

from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from app.events import emit


class AgentState(TypedDict):
    csv_path: str
    csv_profile: str
    question: str
    mode: str                  # "question" | "eda"
    plan: str
    code: str
    stdout: str
    stderr: str
    attempt: int
    max_attempts: int
    race_n: int                # >1 enables the rival-coder race on attempt 1
    race_report: dict          # strategies/outcomes/winner/reason of the race
    chart_path: str | None     # first chart (kept for compatibility)
    charts: list[str]          # all persisted chart paths, in order
    metrics: list[dict]        # structured key metrics parsed from METRICS_JSON
    lessons_used: list[str]    # symptoms of learned lessons injected into the coder
    lessons_learned: int       # new lessons distilled and stored by this run
    reviews_used: int          # quality-revision budget consumed
    verified: bool             # summary fact-checked against stdout
    result_summary: str
    history: list[dict]        # every attempt: {code, stderr, critique}
    status: str


def _route_after_plan(state: AgentState) -> str:
    return "race" if state.get("race_n", 1) > 1 else "write_code"


def _route_after_run(state: AgentState) -> str:
    return "review" if state["status"] == "reviewing" else "critique"


def _route_after_review(state: AgentState) -> str:
    return "write_code" if state["status"] == "fixing" else "summarize"


def _route_after_critique(state: AgentState) -> str:
    # exhausted runs still visit the librarian — failures are worth learning from
    return "write_code" if state["status"] == "fixing" else "learn"


def _instrument(name: str, fn: Callable) -> Callable:
    """Emit node lifecycle events so the cockpit can animate the live graph."""

    def wrapped(state: AgentState) -> AgentState:
        emit("node_start", node=name, attempt=state.get("attempt", 0))
        out = fn(state)
        emit("node_end", node=name, status=out.get("status", ""))
        return out

    wrapped.__name__ = name
    return wrapped


def build_graph():
    from app import nodes

    graph = StateGraph(AgentState)

    for name, fn in [
        ("profile_csv", nodes.profile_csv),
        ("plan", nodes.plan),
        ("write_code", nodes.write_code),
        ("race", nodes.race),
        ("execute", nodes.execute),
        ("review", nodes.review),
        ("critique", nodes.critique),
        ("summarize", nodes.summarize),
        ("verify", nodes.verify),
        ("learn", nodes.learn),
    ]:
        graph.add_node(name, _instrument(name, fn))

    graph.set_entry_point("profile_csv")
    graph.add_edge("profile_csv", "plan")
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"race": "race", "write_code": "write_code"},
    )
    graph.add_edge("write_code", "execute")

    # race and execute both end in a run result; the same status routing applies
    graph.add_conditional_edges(
        "execute",
        _route_after_run,
        {"review": "review", "critique": "critique"},
    )
    graph.add_conditional_edges(
        "race",
        _route_after_run,
        {"review": "review", "critique": "critique"},
    )
    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {"write_code": "write_code", "summarize": "summarize"},
    )
    graph.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"write_code": "write_code", "learn": "learn"},
    )
    graph.add_edge("summarize", "verify")
    graph.add_edge("verify", "learn")
    graph.add_edge("learn", END)

    return graph.compile()
