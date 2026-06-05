from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd

from src.data_loader import load_sales_data
from src.embedding_rag_pipeline import build_chroma_retriever_from_file
from src.llm_agent import run_agent
from src.evaluation import evaluate_agent_response


@dataclass
class AgentEvaluationConfig:
    data_path: str = "data/cleaned_sales.csv"
    context_path: str = "data/business_context.md"
    output_path: str = "outputs/agent_evaluation_report.csv"


DEFAULT_TEST_QUESTIONS = [
    {
        "question_id": "Q001",
        "question": "Which category had the highest sales uplift?",
        "expected_tool_signal": "calculate_sales_uplift",
        "question_type": "kpi_analysis",
    },
    {
        "question_id": "Q002",
        "question": "Which region had the strongest promotion ROI?",
        "expected_tool_signal": "calculate_promotion_roi",
        "question_type": "kpi_analysis",
    },
    {
        "question_id": "Q003",
        "question": "Show total sales by category.",
        "expected_tool_signal": "sql_sales_by_category",
        "question_type": "sql_analysis",
    },
    {
        "question_id": "Q004",
        "question": "Show monthly sales trend.",
        "expected_tool_signal": "sql_monthly_sales_trend",
        "question_type": "sql_analysis",
    },
    {
        "question_id": "Q005",
        "question": "Compare campaign performance.",
        "expected_tool_signal": "sql_campaign_performance",
        "question_type": "sql_analysis",
    },
    {
        "question_id": "Q006",
        "question": "Explain what promotion ROI means in this business context.",
        "expected_tool_signal": "rag_retrieval",
        "question_type": "rag_context",
    },
    {
        "question_id": "Q007",
        "question": "Why should we separate promotional impact from organic demand growth?",
        "expected_tool_signal": "rag_retrieval",
        "question_type": "rag_context",
    },
    {
        "question_id": "Q008",
        "question": "Are there any abnormal sales spikes?",
        "expected_tool_signal": "detect_anomalies",
        "question_type": "anomaly_detection",
    },
]


def check_expected_tool_used(tools_used: list[str], expected_tool_signal: str) -> bool:
    tools_text = " | ".join(tools_used)
    return expected_tool_signal in tools_text


def summarize_single_evaluation(
    question_id: str,
    question: str,
    question_type: str,
    expected_tool_signal: str,
    agent_output: dict,
    evaluation_report: pd.DataFrame,
) -> dict:
    tools_used = agent_output.get("tools_used", [])
    data_result = agent_output.get("data_result", None)
    retrieved_chunks = agent_output.get("retrieved_chunks", [])

    metric_map = {
        row["metric"]: row["value"]
        for _, row in evaluation_report.iterrows()
    }

    expected_tool_used = check_expected_tool_used(
        tools_used=tools_used,
        expected_tool_signal=expected_tool_signal,
    )

    output_rows = len(data_result) if isinstance(data_result, pd.DataFrame) else 0

    return {
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "task_type_detected": agent_output.get("task_type"),
        "expected_tool_signal": expected_tool_signal,
        "tools_used": " | ".join(tools_used),
        "expected_tool_used": expected_tool_used,
        "retrieved_chunk_count": len(retrieved_chunks),
        "data_output_rows": output_rows,
        "retrieval_relevance": metric_map.get("retrieval_relevance"),
        "tool_correctness": metric_map.get("tool_correctness"),
        "answer_completeness": metric_map.get("answer_completeness"),
        "hallucination_risk": metric_map.get("hallucination_risk"),
        "latency_seconds": metric_map.get("latency_seconds"),
    }


def run_agent_evaluation(
    config: AgentEvaluationConfig | None = None,
    test_questions: list[dict] | None = None,
) -> pd.DataFrame:
    if config is None:
        config = AgentEvaluationConfig()

    if test_questions is None:
        test_questions = DEFAULT_TEST_QUESTIONS

    if not os.path.exists(config.data_path):
        raise FileNotFoundError(
            f"Cleaned data not found: {config.data_path}. Run `python run_pipeline.py` first."
        )

    if not os.path.exists(config.context_path):
        raise FileNotFoundError(f"Context file not found: {config.context_path}")

    df = load_sales_data(config.data_path)

    retriever = build_chroma_retriever_from_file(
        file_path=config.context_path,
        persist_directory="chroma_db",
        reset_collection=False,
    )

    rows = []

    for item in test_questions:
        start_time = time.time()

        agent_output = run_agent(
            question=item["question"],
            df=df,
            retriever=retriever,
        )

        end_time = time.time()

        evaluation_report = evaluate_agent_response(
            agent_output=agent_output,
            start_time=start_time,
            end_time=end_time,
        )

        row = summarize_single_evaluation(
            question_id=item["question_id"],
            question=item["question"],
            question_type=item["question_type"],
            expected_tool_signal=item["expected_tool_signal"],
            agent_output=agent_output,
            evaluation_report=evaluation_report,
        )

        rows.append(row)

    evaluation_summary = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    evaluation_summary.to_csv(config.output_path, index=False)

    return evaluation_summary


if __name__ == "__main__":
    result = run_agent_evaluation()
    print("Agent evaluation completed.")
    print(result)