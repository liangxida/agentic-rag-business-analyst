from __future__ import annotations

from typing import Dict, Any, Protocol

import pandas as pd

from src.analysis_tools import (
    calculate_sales_uplift,
    calculate_promotion_roi,
    summarize_by_category,
    detect_anomalies,
)

from src.sql_tools import select_sql_tool, run_selected_sql_tool

class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int = 3):
        ...


def classify_question(question: str) -> str:
    """
    Classify the user question into one of the supported task types.

    Supported types:
    - sql
    - roi
    - uplift
    - category
    - anomaly
    - context
    - mixed
    """

    q = question.lower()

    has_context_signal = any(
        keyword in q
        for keyword in [
            "explain",
            "why",
            "business",
            "context",
            "meaning",
            "definition",
            "use case",
        ]
    )

    has_data_signal = any(
        keyword in q
        for keyword in [
            "roi",
            "uplift",
            "category",
            "region",
            "highest",
            "lowest",
            "sales",
            "promotion",
            "campaign",
            "channel",
            "monthly",
            "month",
            "trend",
            "sql",
            "query",
            "anomaly",
            "abnormal",
            "spike",
        ]
    )

    sql_tool = select_sql_tool(question)

    if sql_tool is not None and not ("roi" in q or "uplift" in q or "anomaly" in q):
        if has_context_signal:
            return "mixed"
        return "sql"

    if has_context_signal and has_data_signal:
        return "mixed"

    if "roi" in q:
        return "roi"

    if "uplift" in q:
        return "uplift"

    if "anomaly" in q or "abnormal" in q or "spike" in q:
        return "anomaly"

    if "category" in q or "sales" in q or "promotion" in q:
        return "category"

    return "context"


def dataframe_to_brief_text(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Convert a DataFrame into a short text summary for display.
    """

    if df.empty:
        return "No matching records were found."

    preview = df.head(max_rows).to_string(index=False)
    return preview


def generate_rule_based_answer(
    question: str,
    task_type: str,
    data_result: pd.DataFrame | None,
    retrieved_context: str | None,
) -> str:
    """
    Generate a deterministic answer without calling an external LLM.

    This keeps the MVP runnable without API keys.
    """

    answer_parts = []

    answer_parts.append(f"Question: {question}")
    answer_parts.append(f"Detected task type: {task_type}")

    if retrieved_context:
        answer_parts.append("\nRelevant business context:")
        answer_parts.append(retrieved_context)

    if data_result is not None:
        answer_parts.append("\nData-backed result:")
        answer_parts.append(dataframe_to_brief_text(data_result))

        if not data_result.empty:
            first_row = data_result.iloc[0].to_dict()

            if task_type in ["roi", "mixed"] and "promotion_roi" in data_result.columns:
                answer_parts.append(
                    "\nInterpretation: The top row shows the segment with the highest estimated promotion ROI. "
                    "This indicates where incremental sales were strongest relative to promotion cost."
                )

            elif task_type in ["uplift", "mixed"] and "sales_uplift" in data_result.columns:
                answer_parts.append(
                    "\nInterpretation: The top row shows the segment with the highest estimated sales uplift. "
                    "This means actual promoted sales exceeded estimated baseline sales by the largest amount."
                )

            elif task_type == "category":
                answer_parts.append(
                    "\nInterpretation: This table summarizes category-level sales, promotion usage, and estimated uplift."
                )

            elif task_type == "anomaly":
                answer_parts.append(
                    "\nInterpretation: These rows show unusually high or low sales compared with the category-level average."
                )

    answer_parts.append(
        "\nNote: This MVP uses a rule-based response generator. In a production version, this layer can be replaced with an LLM to generate more natural business explanations."
    )

    return "\n".join(answer_parts)


def run_agent(
    question: str,
    df: pd.DataFrame,
    retriever: RetrieverProtocol,
) -> Dict[str, Any]:
    """
    Main agent workflow.

    It decides whether to call KPI tools, RAG retrieval, or both.
    """

    task_type = classify_question(question)

    data_result = None
    retrieved_chunks = []
    retrieved_context = None
    tools_used = []

    if task_type == "sql":
        sql_tool = select_sql_tool(question)
        if sql_tool is not None:
            data_result = run_selected_sql_tool(sql_tool)
            tools_used.append(f"sql_tool:{sql_tool}")

    elif task_type == "mixed":
        sql_tool = select_sql_tool(question)

        if "roi" in question.lower():
            data_result = calculate_promotion_roi(df)
            tools_used.append("calculate_promotion_roi")
        elif "uplift" in question.lower():
            data_result = calculate_sales_uplift(df)
            tools_used.append("calculate_sales_uplift")
        elif "anomaly" in question.lower() or "abnormal" in question.lower() or "spike" in question.lower():
            data_result = detect_anomalies(df)
            tools_used.append("detect_anomalies")
        elif sql_tool is not None:
            data_result = run_selected_sql_tool(sql_tool)
            tools_used.append(f"sql_tool:{sql_tool}")
        else:
            data_result = summarize_by_category(df)
            tools_used.append("summarize_by_category")

    elif task_type == "roi":
        data_result = calculate_promotion_roi(df)
        tools_used.append("calculate_promotion_roi")

    elif task_type == "uplift":
        data_result = calculate_sales_uplift(df)
        tools_used.append("calculate_sales_uplift")

    elif task_type == "category":
        data_result = summarize_by_category(df)
        tools_used.append("summarize_by_category")

    elif task_type == "anomaly":
        data_result = detect_anomalies(df)
        tools_used.append("detect_anomalies")
        
    if task_type in ["context", "mixed"]:
        retrieved_chunks = retriever.retrieve(question, top_k=3)
        retrieved_context = "\n\n".join(
            [
                f"[Chunk {chunk.chunk_id}, score={chunk.score:.3f}]\n{chunk.text}"
                for chunk in retrieved_chunks
            ]
        )
        tools_used.append("rag_retrieval")

    answer = generate_rule_based_answer(
        question=question,
        task_type=task_type,
        data_result=data_result,
        retrieved_context=retrieved_context,
    )

    return {
        "question": question,
        "task_type": task_type,
        "tools_used": tools_used,
        "data_result": data_result,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
    }