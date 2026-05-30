from __future__ import annotations

import os
import sqlite3
from typing import Optional

import pandas as pd


DEFAULT_DB_PATH = "data/business_analytics.db"
DEFAULT_TABLE_NAME = "sales"


def create_sqlite_database(
    csv_path: str = "data/sample_sales.csv",
    db_path: str = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
) -> None:
    """
    Create a SQLite database from the sample sales CSV.

    This allows the agent to call SQL tools over structured business data.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def run_sql_query(
    query: str,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Run a read-only SQL query against the SQLite database.

    For safety, only SELECT queries are allowed.
    """

    cleaned_query = query.strip().lower()

    if not cleaned_query.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    blocked_keywords = ["drop", "delete", "insert", "update", "alter", "create"]
    if any(keyword in cleaned_query for keyword in blocked_keywords):
        raise ValueError("Unsafe SQL keyword detected.")

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def get_total_sales_by_category(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    SQL tool: total sales by category.
    """

    query = """
    SELECT
        category,
        ROUND(SUM(sales), 2) AS total_sales,
        SUM(units_sold) AS total_units,
        ROUND(AVG(discount_rate), 3) AS avg_discount
    FROM sales
    GROUP BY category
    ORDER BY total_sales DESC;
    """

    return run_sql_query(query, db_path=db_path)


def get_total_sales_by_region(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    SQL tool: total sales by region.
    """

    query = """
    SELECT
        region,
        ROUND(SUM(sales), 2) AS total_sales,
        SUM(units_sold) AS total_units,
        ROUND(AVG(discount_rate), 3) AS avg_discount
    FROM sales
    GROUP BY region
    ORDER BY total_sales DESC;
    """

    return run_sql_query(query, db_path=db_path)


def get_campaign_performance(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    SQL tool: campaign-level performance.
    """

    query = """
    SELECT
        campaign_name,
        COUNT(*) AS row_count,
        ROUND(SUM(sales), 2) AS total_sales,
        SUM(units_sold) AS total_units,
        ROUND(SUM(promotion_cost), 2) AS total_promotion_cost,
        ROUND(AVG(discount_rate), 3) AS avg_discount
    FROM sales
    WHERE promotion_flag = 1
    GROUP BY campaign_name
    ORDER BY total_sales DESC;
    """

    return run_sql_query(query, db_path=db_path)


def get_channel_performance(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    SQL tool: channel-level performance.
    """

    query = """
    SELECT
        channel,
        ROUND(SUM(sales), 2) AS total_sales,
        SUM(units_sold) AS total_units,
        ROUND(AVG(promotion_flag), 3) AS promotion_rate,
        ROUND(AVG(discount_rate), 3) AS avg_discount
    FROM sales
    GROUP BY channel
    ORDER BY total_sales DESC;
    """

    return run_sql_query(query, db_path=db_path)


def get_monthly_sales_trend(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    SQL tool: monthly sales trend.
    """

    query = """
    SELECT
        SUBSTR(date, 1, 7) AS month,
        ROUND(SUM(sales), 2) AS total_sales,
        SUM(units_sold) AS total_units,
        ROUND(AVG(promotion_flag), 3) AS promotion_rate
    FROM sales
    GROUP BY SUBSTR(date, 1, 7)
    ORDER BY month;
    """

    return run_sql_query(query, db_path=db_path)


def select_sql_tool(question: str) -> Optional[str]:
    """
    Select a predefined SQL tool based on the user question.
    """

    q = question.lower()

    if "campaign" in q:
        return "campaign_performance"

    if "channel" in q:
        return "channel_performance"

    if "monthly" in q or "month" in q or "trend" in q:
        return "monthly_sales_trend"

    if "region" in q:
        return "sales_by_region"

    if "category" in q:
        return "sales_by_category"

    return None


def run_selected_sql_tool(
    tool_name: str,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Run a predefined SQL tool by name.
    """

    tool_map = {
        "sales_by_category": get_total_sales_by_category,
        "sales_by_region": get_total_sales_by_region,
        "campaign_performance": get_campaign_performance,
        "channel_performance": get_channel_performance,
        "monthly_sales_trend": get_monthly_sales_trend,
    }

    if tool_name not in tool_map:
        raise ValueError(f"Unknown SQL tool: {tool_name}")

    return tool_map[tool_name](db_path=db_path)