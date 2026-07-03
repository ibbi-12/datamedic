"""LangGraph build: state schema, node wiring, and edges."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    csv_path: str
    csv_profile: str
    question: str
    plan: str
    code: str
    stdout: str
    stderr: str
    attempt: int
    max_attempts: int
    chart_path: str | None
    result_summary: str
    history: list[dict]
    status: str


def _route_after_execute(state: AgentState) -> str:
    return "summarize" if state["status"] == "summarizing" else "critique"


def _route_after_critique(state: AgentState) -> str:
    return "write_code" if state["status"] == "fixing" else "end_failed"


def build_graph():
    from app import nodes

    graph = StateGraph(AgentState)

    graph.add_node("profile_csv", nodes.profile_csv)
    graph.add_node("plan", nodes.plan)
    graph.add_node("write_code", nodes.write_code)
    graph.add_node("execute", nodes.execute)
    graph.add_node("critique", nodes.critique)
    graph.add_node("summarize", nodes.summarize)

    graph.set_entry_point("profile_csv")
    graph.add_edge("profile_csv", "plan")
    graph.add_edge("plan", "write_code")
    graph.add_edge("write_code", "execute")

    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {"summarize": "summarize", "critique": "critique"},
    )
    graph.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"write_code": "write_code", "end_failed": END},
    )
    graph.add_edge("summarize", END)

    return graph.compile()
