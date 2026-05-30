# Agentic RAG Business Analyst

## Overview

Agentic RAG Business Analyst is an AI-powered business analytics prototype that combines SQL tool calling, Python-based KPI analysis, embedding-based document retrieval, and an agentic workflow to answer business questions from structured CSV data and business context documents.

The project simulates a retail promotion analytics use case where business users need to evaluate promotion ROI, sales uplift, category performance, regional performance, campaign performance, channel performance, and abnormal sales patterns.

## Business Problem

Business teams often need to determine whether promotions generate incremental sales beyond organic demand growth. Raw sales alone can be misleading because sales may be affected by seasonality, regional demand differences, baseline product popularity, campaign design, or channel mix.

This project provides an agentic business analyst interface that routes natural language questions to the appropriate analytical tools, SQL tools, or retrieval pipeline.

## Key Features

- Agentic workflow for business question routing
- SQL tool calling over structured sales data
- Controlled predefined SQL tools to reduce unsafe query execution risk
- Promotion ROI calculation
- Sales uplift estimation against baseline sales
- Category-level, region-level, channel-level, campaign-level, and monthly trend analysis
- Data validation and quality checks
- Anomaly detection for abnormal sales spikes
- Document chunking
- Embedding-based retrieval
- Chroma vector database
- Semantic search over business context documents
- Streamlit dashboard for business users
- Evaluation report covering tool usage, retrieval relevance, answer completeness, hallucination risk, and latency

## Screenshots

### SQL Tool Calling

![SQL Tool Calling](screenshots/02_sql_tool_calling.png)

### RAG Retrieval

![RAG Retrieval](screenshots/03_rag_retrieval.png)

### Evaluation Report

![Evaluation Report](screenshots/04_evaluation_report.png)

## Tech Stack

- Python
- SQL / SQLite
- pandas
- Streamlit
- SentenceTransformers
- Chroma vector database
- scikit-learn
- Rule-based agent routing
- Controlled SQL tool calling

## Project Structure

```text
agentic-rag-business-analyst/
  app.py
  README.md
  requirements.txt
  data/
    sample_sales.csv
    business_context.md
    business_analytics.db
  src/
    __init__.py
    generate_sample_data.py
    data_loader.py
    analysis_tools.py
    embedding_rag_pipeline.py
    sql_tools.py
    llm_agent.py
    evaluation.py
  screenshots/








