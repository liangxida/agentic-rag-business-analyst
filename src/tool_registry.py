from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import pandas as pd

from src.analysis_tools import (
    calculate_sales_uplift,
    calculate_promotion_roi,
    summarize_by_category,
    detect_anomalies,
)
from src.sql_tools import (
    get_total_sales_by_category,
    get_total_sales_by_region,
    get_campaign_performance,
    get_channel_performance,
    get_monthly_sales_trend,
)


@dataclass
class ToolSpec:
    """
    Metadata for a registered agent tool.
    """

    name: str
    description: str
    category: str
    input_type: str
    output_type: str
    function: Callable[..., Any]


class ToolRegistry:
    """
    Plugin-style tool registry for the agent.

    This registry makes SQL tools, KPI tools, and future RAG/evaluation tools
    discoverable and extensible.
    """

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}

    def register(self, tool_spec: ToolSpec) -> None:
        if tool_spec.name in self.tools:
            raise ValueError(f"Tool already registered: {tool_spec.name}")

        self.tools[tool_spec.name] = tool_spec

    def get_tool(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")

        return self.tools[name]

    def list_tools(self) -> pd.DataFrame:
        rows = []

        for tool in self.tools.values():
            rows.append(
                {
                    "name": tool.name,
                    "category": tool.category,
                    "input_type": tool.input_type,
                    "output_type": tool.output_type,
                    "description": tool.description,
                }
            )

        return pd.DataFrame(rows).sort_values(["category", "name"])

    def run_tool(self, name: str, **kwargs) -> Any:
        tool = self.get_tool(name)
        return tool.function(**kwargs)


def build_default_tool_registry() -> ToolRegistry:
    """
    Build the default plugin-style tool registry.
    """

    registry = ToolRegistry()

    # KPI / Python analytical tools
    registry.register(
        ToolSpec(
            name="calculate_sales_uplift",
            description="Estimate promotion-driven sales uplift using baseline sales from non-promotion periods.",
            category="kpi_analysis",
            input_type="dataframe",
            output_type="dataframe",
            function=calculate_sales_uplift,
        )
    )

    registry.register(
        ToolSpec(
            name="calculate_promotion_roi",
            description="Calculate promotion ROI by comparing estimated sales uplift against promotion cost.",
            category="kpi_analysis",
            input_type="dataframe",
            output_type="dataframe",
            function=calculate_promotion_roi,
        )
    )

    registry.register(
        ToolSpec(
            name="summarize_by_category",
            description="Summarize total sales, units, promotion rate, discount, and uplift by product category.",
            category="kpi_analysis",
            input_type="dataframe",
            output_type="dataframe",
            function=summarize_by_category,
        )
    )

    registry.register(
        ToolSpec(
            name="detect_anomalies",
            description="Detect abnormal sales records using z-score based anomaly detection within each category.",
            category="kpi_analysis",
            input_type="dataframe",
            output_type="dataframe",
            function=detect_anomalies,
        )
    )

    # SQL tools
    registry.register(
        ToolSpec(
            name="sql_sales_by_category",
            description="Query total sales and units by product category from SQLite.",
            category="sql_analysis",
            input_type="sqlite",
            output_type="dataframe",
            function=get_total_sales_by_category,
        )
    )

    registry.register(
        ToolSpec(
            name="sql_sales_by_region",
            description="Query total sales and units by region from SQLite.",
            category="sql_analysis",
            input_type="sqlite",
            output_type="dataframe",
            function=get_total_sales_by_region,
        )
    )

    registry.register(
        ToolSpec(
            name="sql_campaign_performance",
            description="Query campaign-level sales, units, promotion cost, and average discount.",
            category="sql_analysis",
            input_type="sqlite",
            output_type="dataframe",
            function=get_campaign_performance,
        )
    )

    registry.register(
        ToolSpec(
            name="sql_channel_performance",
            description="Query channel-level sales, units, promotion rate, and average discount.",
            category="sql_analysis",
            input_type="sqlite",
            output_type="dataframe",
            function=get_channel_performance,
        )
    )

    registry.register(
        ToolSpec(
            name="sql_monthly_sales_trend",
            description="Query monthly total sales, units, and promotion rate.",
            category="sql_analysis",
            input_type="sqlite",
            output_type="dataframe",
            function=get_monthly_sales_trend,
        )
    )

    return registry