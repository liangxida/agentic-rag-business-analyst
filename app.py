from __future__ import annotations

import os

import streamlit as st
import pandas as pd


from src.data_loader import (
    load_sales_data,
    validate_sales_data,
    summarize_sales_data,
)
from src.embedding_rag_pipeline import build_chroma_retriever_from_file
from src.llm_agent import run_agent
from src.evaluation import evaluate_agent_response, current_time
from src.sql_tools import create_sqlite_database
from src.tool_registry import build_default_tool_registry

st.set_page_config(
    page_title="Agentic RAG Business Analyst",
    page_icon="📊",
    layout="wide",
)


st.title("Agentic RAG Business Analyst")
st.write(
    """
    This prototype combines structured KPI analysis, lightweight RAG retrieval,
    and an agentic workflow to answer business questions from CSV data and business documents.
    """
)


DEFAULT_DATA_PATH = "data/cleaned_sales.csv"
DEFAULT_CONTEXT_PATH = "data/business_context.md"


@st.cache_data
def cached_load_data(path: str) -> pd.DataFrame:
    return load_sales_data(path)


@st.cache_resource
def cached_build_retriever(path: str):
    return build_chroma_retriever_from_file(
        file_path=path,
        persist_directory="chroma_db",
        reset_collection=False,
    )


with st.sidebar:
    st.header("Project Settings")

    data_path = st.text_input("Sales CSV path", value=DEFAULT_DATA_PATH)
    context_path = st.text_input("Business context path", value=DEFAULT_CONTEXT_PATH)

    st.markdown("---")
    st.write("Example questions:")

    st.code("Which category had the highest sales uplift?")
    st.code("Which region had the strongest promotion ROI?")
    st.code("Explain what promotion ROI means in this business context.")
    st.code("Are there any abnormal sales spikes?")
    st.code("Why should we separate promotional impact from organic demand growth?")
    st.code("Show total sales by category.")
    st.code("Show monthly sales trend.")
    st.code("Compare campaign performance.")
    st.code("Which channel generated the highest sales?")


if not os.path.exists(data_path):
    st.error(
        f"Data file not found: {data_path}. Please run src/generate_sample_data.py first."
    )
    st.stop()

if not os.path.exists(context_path):
    st.error(f"Context file not found: {context_path}. Please create business_context.md.")
    st.stop()

df = cached_load_data(data_path)
retriever = cached_build_retriever(context_path)
create_sqlite_database(csv_path=data_path)



tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Ask Agent",
        "Data Overview",
        "Data Quality",
        "ETL Pipeline",
        "Tool Registry",
        "Project Notes",
    ]
)

with tab1:
    st.subheader("Ask a Business Question")

    question = st.text_input(
        "Enter your question",
        value="Which category had the highest sales uplift?",
    )

    run_button = st.button("Run Agent")

    if run_button:
        start_time = current_time()
        agent_output = run_agent(question=question, df=df, retriever=retriever)
        end_time = current_time()

        evaluation_report = evaluate_agent_response(
            agent_output=agent_output,
            start_time=start_time,
            end_time=end_time,
        )

        st.markdown("### Agent Answer")
        st.text(agent_output["answer"])

        st.markdown("### Tools Used")
        st.write(agent_output["tools_used"])

        if agent_output["data_result"] is not None:
            st.markdown("### Data Result")
            st.dataframe(agent_output["data_result"], use_container_width=True)

        if agent_output["retrieved_chunks"]:
            st.markdown("### Retrieved Context")
            for chunk in agent_output["retrieved_chunks"]:
                with st.expander(f"Chunk {chunk.chunk_id} | Score: {chunk.score:.3f}"):
                    st.write(chunk.text)

        st.markdown("### Evaluation Report")
        st.dataframe(evaluation_report, use_container_width=True)


with tab2:
    st.subheader("Sales Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Category Summary")
    summary = summarize_sales_data(df)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Total Sales by Category")
    chart_data = summary.set_index("category")["total_sales"]
    st.bar_chart(chart_data)


with tab3:
    st.subheader("Data Quality")

    st.write(
        """
        This tab shows both in-memory data validation results and the ETL-generated
        data quality report used to prepare cleaned data for downstream agent tools.
        """
    )

    st.markdown("### In-memory Data Quality Report")

    quality_report = validate_sales_data(df)
    st.dataframe(quality_report, use_container_width=True)

    st.markdown("### ETL-generated Data Quality Report")

    quality_report_path = "outputs/data_quality_report.csv"

    if os.path.exists(quality_report_path):
        etl_quality_report = pd.read_csv(quality_report_path)
        st.dataframe(etl_quality_report, use_container_width=True)

        st.markdown("### Data Quality Status Summary")

        status_summary = (
            etl_quality_report.groupby("status")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        st.dataframe(status_summary, use_container_width=True)

        st.markdown("### Rows Before and After Cleaning")

        row_summary = etl_quality_report[
            etl_quality_report["check_name"].isin(
                ["rows_before_cleaning", "rows_after_cleaning"]
            )
        ][["check_name", "result"]]

        st.dataframe(row_summary, use_container_width=True)

    else:
        st.warning("ETL data quality report not found. Run `python run_pipeline.py` first.")

    st.markdown("### KPI-ready Feature Columns")

    engineered_columns = [
        "month",
        "week",
        "revenue_per_unit",
        "is_discounted",
        "promotion_intensity",
        "sales_per_promotion_dollar",
    ]

    existing_engineered_columns = [col for col in engineered_columns if col in df.columns]

    if existing_engineered_columns:
        st.write(existing_engineered_columns)
        st.dataframe(df[existing_engineered_columns].head(10), use_container_width=True)
    else:
        st.warning("No engineered KPI feature columns found in the current dataset.")


with tab4:
    st.subheader("ETL Pipeline and Automated Agent Evaluation")

    st.write(
        """
        This tab summarizes the automated pipeline and evaluates whether the agent
        selects the expected tools for predefined business questions.
        """
    )

    st.markdown("### ETL Pipeline Flow")

    st.code(
        """
Raw sales CSV
  → schema validation
  → data quality checks
  → cleaning and type standardization
  → KPI-ready feature transformation
  → cleaned CSV export
  → SQLite database refresh
  → downstream SQL and KPI tools
        """
    )

    st.markdown("### ETL Outputs")
    st.write("- `data/cleaned_sales.csv`")
    st.write("- `outputs/data_quality_report.csv`")
    st.write("- `outputs/agent_evaluation_report.csv`")
    st.write("- `data/business_analytics.db`")

    st.markdown("### Automated Agent Evaluation Report")

    agent_eval_path = "outputs/agent_evaluation_report.csv"

    if os.path.exists(agent_eval_path):
        agent_eval_report = pd.read_csv(agent_eval_path)
        st.dataframe(agent_eval_report, use_container_width=True)

        if "expected_tool_used" in agent_eval_report.columns:
            expected_tool_accuracy = agent_eval_report["expected_tool_used"].mean()

            st.metric(
                label="Expected Tool Usage Accuracy",
                value=f"{expected_tool_accuracy:.0%}",
            )

        if "latency_seconds" in agent_eval_report.columns:
            latency_numeric = pd.to_numeric(
                agent_eval_report["latency_seconds"],
                errors="coerce",
            )

            st.metric(
                label="Average Latency",
                value=f"{latency_numeric.mean():.3f}s",
            )

        if "hallucination_risk" in agent_eval_report.columns:
            risk_summary = (
                agent_eval_report.groupby("hallucination_risk")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )

            st.markdown("### Hallucination Risk Summary")
            st.dataframe(risk_summary, use_container_width=True)

    else:
        st.warning("Agent evaluation report not found. Run `python run_pipeline.py` first.")

with tab5:
    st.subheader("Plugin-style Tool Registry")

    st.write(
        """
        The agent uses a plugin-style tool registry to organize SQL tools,
        KPI analysis tools, and other modular skills. This design makes the
        agent workflow easier to extend with new tools.
        """
    )

    tool_registry = build_default_tool_registry()
    tool_table = tool_registry.list_tools()

    st.markdown("### Registered Tools")
    
    st.dataframe(tool_table, use_container_width=True)

    st.markdown("### Tool Categories")
    category_summary = (
        tool_table.groupby("category")
        .size()
        .reset_index(name="tool_count")
        .sort_values("tool_count", ascending=False)
    )
    st.dataframe(category_summary, use_container_width=True)

    st.markdown("### Why this matters")
    st.write(
        """
        Instead of hard-coding every action inside a single function, tools are
        registered with metadata including name, category, input type, output type,
        and description. This approximates a skills/plugins framework for applied
        AI agents.
        """
    )

with tab6:
    st.subheader("Project Notes")

    st.write(
        """
        This project is designed to demonstrate practical Applied AI Data Science skills,
        including RAG, SQL tool calling, ETL automation, data quality validation,
        plugin-style tool registry, KPI analysis, and agent evaluation.
        """
    )

    st.markdown(
        """
        **Covered concepts:**

        - Agentic workflow
        - Retrieval-Augmented Generation
        - Document chunking
        - Embedding-based retrieval
        - Vector database with Chroma
        - Semantic search
        - Retrieval scoring
        - SQL tool calling
        - Plugin-style skills framework
        - Modular tool registry
        - ETL pipeline
        - Data validation
        - Data quality reporting
        - Automated agent evaluation
        - KPI analysis
        - Promotion ROI
        - Sales uplift
        - Hallucination risk tracking
        - Latency tracking
        - Streamlit business prototype
        """
    )