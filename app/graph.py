"""LangGraph build: state schema, node wiring, and edges."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph


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
    chart_path: str | None     # first chart (kept for compatibility)
    charts: list[str]          # all persisted chart paths, in order
    metrics: list[dict]        # structured key metrics parsed from METRICS_JSON
    reviews_used: int          # quality-revision budget consumed
    verified: bool             # summary fact-checked against stdout
    result_summary: str
    history: list[dict]        # every attempt: {code, stderr, critique}
    status: str


def _route_after_execute(state: AgentState) -> str:
    return "review" if state["status"] == "reviewing" else "critique"


def _route_after_review(state: AgentState) -> str:
    return "write_code" if state["status"] == "fixing" else "summarize"


def _route_after_critique(state: AgentState) -> str:
    return "write_code" if state["status"] == "fixing" else "end_failed"


def build_graph():
    from app import nodes

    graph = StateGraph(AgentState)

    graph.add_node("profile_csv", nodes.profile_csv)
    graph.add_node("plan", nodes.plan)
    graph.add_node("write_code", nodes.write_code)
    graph.add_node("execute", nodes.execute)
    graph.add_node("review", nodes.review)
    graph.add_node("critique", nodes.critique)
    graph.add_node("summarize", nodes.summarize)
    graph.add_node("verify", nodes.verify)

    graph.set_entry_point("profile_csv")
    graph.add_edge("profile_csv", "plan")
    graph.add_edge("plan", "write_code")
    graph.add_edge("write_code", "execute")

    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
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
        {"write_code": "write_code", "end_failed": END},
    )
    graph.add_edge("summarize", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
