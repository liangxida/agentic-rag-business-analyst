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


DEFAULT_DATA_PATH = "data/sample_sales.csv"
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



tab1, tab2, tab3, tab4 = st.tabs(
    ["Ask Agent", "Data Overview", "Data Quality", "Project Notes"]
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
    st.subheader("Data Quality Report")
    quality_report = validate_sales_data(df)
    st.dataframe(quality_report, use_container_width=True)


with tab4:
    st.subheader("Project Positioning")
    st.write(
        """
        This project is designed to demonstrate practical AI/data skills for GenAI,
        AI agent, product analytics, and data analyst roles.
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
        - Tool calling
        - KPI analysis
        - Promotion ROI
        - Sales uplift
        - Data validation
        - LLM evaluation framework
        - Streamlit business prototype
        """
    )