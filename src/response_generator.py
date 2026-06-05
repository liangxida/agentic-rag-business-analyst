from __future__ import annotations

from typing import Optional

import pandas as pd


def dataframe_to_markdown_preview(df: Optional[pd.DataFrame], max_rows: int = 5) -> str:
    """
    Convert a DataFrame result into a compact markdown preview.
    """

    if df is None:
        return "No structured data result was produced."

    if df.empty:
        return "The tool returned an empty result."

    return df.head(max_rows).to_markdown(index=False)


def build_grounded_prompt(
    question: str,
    task_type: str,
    tools_used: list[str],
    data_result: Optional[pd.DataFrame],
    retrieved_context: Optional[str],
) -> str:
    """
    Build a source-grounded prompt for an LLM or deterministic response generator.
    """

    data_preview = dataframe_to_markdown_preview(data_result)

    context_text = retrieved_context if retrieved_context else "No retrieved context was used."

    tools_text = ", ".join(tools_used) if tools_used else "No tools were used."

    prompt = f"""
You are an applied AI business analyst assistant.

Your task is to answer the user's question using only the provided tool outputs and retrieved context.

Rules:
1. Do not invent facts that are not supported by the tool outputs or retrieved context.
2. If the evidence is insufficient, clearly say what is missing.
3. Explain the result in business language.
4. Mention which tools or evidence were used.
5. Keep the answer concise and decision-oriented.

User Question:
{question}

Detected Task Type:
{task_type}

Tools Used:
{tools_text}

Structured Data Result:
{data_preview}

Retrieved Context:
{context_text}

Write a grounded answer with:
- Direct answer
- Evidence used
- Business interpretation
- Limitation or next step
"""
    return prompt.strip()


def generate_fallback_grounded_answer(
    question: str,
    task_type: str,
    tools_used: list[str],
    data_result: Optional[pd.DataFrame],
    retrieved_context: Optional[str],
) -> str:
    """
    Generate a grounded answer without calling an external LLM.

    This keeps the project runnable while still demonstrating a grounded-response layer.
    """

    tools_text = ", ".join(tools_used) if tools_used else "no tools"

    answer_parts = []

    answer_parts.append("Direct answer:")
    answer_parts.append(
        f"The question was routed as a `{task_type}` task and answered using {tools_text}."
    )

    if data_result is not None:
        if data_result.empty:
            answer_parts.append(
                "The selected analytical tool returned no rows, so there is no data-backed ranking to report."
            )
        else:
            top_row = data_result.iloc[0].to_dict()
            answer_parts.append(
                "The top data-backed result is shown below based on the selected analytical tool:"
            )
            answer_parts.append(str(top_row))

    if retrieved_context:
        answer_parts.append("\nEvidence used:")
        answer_parts.append(
            "The answer was grounded using retrieved business context and/or structured tool outputs."
        )
    else:
        answer_parts.append("\nEvidence used:")
        answer_parts.append(
            "The answer was grounded using structured tool outputs. No document context was required for this question."
        )

    answer_parts.append("\nBusiness interpretation:")

    if "promotion_roi" in " ".join(tools_used) or "roi" in question.lower():
        answer_parts.append(
            "Promotion ROI helps compare incremental sales impact against promotion cost, making it useful for budget allocation and campaign prioritization."
        )
    elif "uplift" in question.lower():
        answer_parts.append(
            "Sales uplift estimates how much promoted sales exceeded baseline demand, helping separate promotional impact from organic growth."
        )
    elif "anomal" in question.lower() or "spike" in question.lower():
        answer_parts.append(
            "Anomaly detection helps identify abnormal sales patterns that may require data quality review or campaign performance investigation."
        )
    elif "trend" in question.lower() or "monthly" in question.lower():
        answer_parts.append(
            "Trend analysis helps business users monitor performance changes over time and connect metric changes to campaign or operational events."
        )
    else:
        answer_parts.append(
            "The result provides a structured view of business performance and can support stakeholder-facing analysis."
        )

    answer_parts.append("\nLimitation or next step:")
    answer_parts.append(
        "This answer is generated from available tool outputs and retrieved context. A production system should add stronger source citation, user feedback logging, and periodic evaluation."
    )

    return "\n".join(answer_parts)


def generate_grounded_answer(
    question: str,
    task_type: str,
    tools_used: list[str],
    data_result: Optional[pd.DataFrame],
    retrieved_context: Optional[str],
    use_external_llm: bool = False,
) -> str:
    """
    Generate a grounded response.

    Current implementation:
    - Builds a grounded prompt.
    - Uses deterministic fallback by default.
    - Can be extended to call an external LLM later.
    """

    _ = build_grounded_prompt(
        question=question,
        task_type=task_type,
        tools_used=tools_used,
        data_result=data_result,
        retrieved_context=retrieved_context,
    )

    # Safe default: no external API call.
    # This makes the project reproducible for GitHub reviewers.
    return generate_fallback_grounded_answer(
        question=question,
        task_type=task_type,
        tools_used=tools_used,
        data_result=data_result,
        retrieved_context=retrieved_context,
    )