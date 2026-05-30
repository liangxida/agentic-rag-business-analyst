from __future__ import annotations

import time
from typing import Dict, Any

import pandas as pd


def evaluate_agent_response(
    agent_output: Dict[str, Any],
    start_time: float,
    end_time: float,
) -> pd.DataFrame:
    """
    Create a simple evaluation report for the agent response.

    This is a lightweight MVP evaluation framework.
    """

    retrieved_chunks = agent_output.get("retrieved_chunks", [])
    tools_used = agent_output.get("tools_used", [])
    data_result = agent_output.get("data_result", None)
    answer = agent_output.get("answer", "")

    retrieval_relevance = None
    if retrieved_chunks:
        retrieval_relevance = sum(chunk.score for chunk in retrieved_chunks) / len(
            retrieved_chunks
        )

    tool_correctness = "not_applicable"
    if data_result is not None:
        tool_correctness = "pass" if isinstance(data_result, pd.DataFrame) else "fail"

    answer_has_content = len(answer.strip()) > 50

    hallucination_risk = "medium"
    if retrieved_chunks or data_result is not None:
        hallucination_risk = "low"

    latency_seconds = round(end_time - start_time, 3)

    rows = [
        {
            "metric": "tools_used",
            "value": ", ".join(tools_used) if tools_used else "none",
            "interpretation": "Which tools were triggered by the agent.",
        },
        {
            "metric": "retrieval_relevance",
            "value": round(retrieval_relevance, 3)
            if retrieval_relevance is not None
            else "not_applicable",
            "interpretation": "Average retrieval similarity score for retrieved context.",
        },
        {
            "metric": "tool_correctness",
            "value": tool_correctness,
            "interpretation": "Whether the selected analytical tool returned a valid result.",
        },
        {
            "metric": "answer_completeness",
            "value": "pass" if answer_has_content else "fail",
            "interpretation": "Whether the generated answer contains enough explanatory content.",
        },
        {
            "metric": "hallucination_risk",
            "value": hallucination_risk,
            "interpretation": "Estimated risk based on whether the answer is grounded in retrieved context or data tools.",
        },
        {
            "metric": "latency_seconds",
            "value": latency_seconds,
            "interpretation": "End-to-end runtime for the agent workflow.",
        },
    ]

    return pd.DataFrame(rows)


def current_time() -> float:
    return time.time()