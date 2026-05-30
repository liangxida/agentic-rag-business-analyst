from __future__ import annotations

import pandas as pd


def estimate_baseline_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate baseline sales by category and region using non-promotion periods.

    This is a simple baseline method for MVP:
    baseline_sales = average sales during non-promotion days for the same category-region.
    """

    non_promo = df[df["promotion_flag"] == 0].copy()

    baseline = (
        non_promo.groupby(["category", "region"])
        .agg(baseline_sales=("sales", "mean"))
        .reset_index()
    )

    result = df.merge(baseline, on=["category", "region"], how="left")

    global_baseline = non_promo["sales"].mean()
    result["baseline_sales"] = result["baseline_sales"].fillna(global_baseline)

    return result


def calculate_sales_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sales uplift for promotion rows.

    sales_uplift = actual sales - estimated baseline sales
    """

    result = estimate_baseline_sales(df)
    result["sales_uplift"] = result["sales"] - result["baseline_sales"]

    promo_result = result[result["promotion_flag"] == 1].copy()

    uplift_summary = (
        promo_result.groupby(["category", "region"])
        .agg(
            actual_sales=("sales", "sum"),
            baseline_sales=("baseline_sales", "sum"),
            sales_uplift=("sales_uplift", "sum"),
            promotion_cost=("promotion_cost", "sum"),
        )
        .reset_index()
    )

    uplift_summary["uplift_rate"] = (
        uplift_summary["sales_uplift"] / uplift_summary["baseline_sales"]
    )

    numeric_cols = [
        "actual_sales",
        "baseline_sales",
        "sales_uplift",
        "promotion_cost",
        "uplift_rate",
    ]

    for col in numeric_cols:
        uplift_summary[col] = uplift_summary[col].round(3)

    return uplift_summary.sort_values("sales_uplift", ascending=False)


def calculate_promotion_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate promotion ROI.

    promotion_roi = sales_uplift / promotion_cost
    """

    uplift_summary = calculate_sales_uplift(df)

    uplift_summary["promotion_roi"] = uplift_summary.apply(
        lambda row: row["sales_uplift"] / row["promotion_cost"]
        if row["promotion_cost"] > 0
        else 0,
        axis=1,
    )

    uplift_summary["promotion_roi"] = uplift_summary["promotion_roi"].round(3)

    return uplift_summary.sort_values("promotion_roi", ascending=False)


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sales and promotion performance by category.
    """

    category_summary = (
        df.groupby("category")
        .agg(
            total_sales=("sales", "sum"),
            total_units=("units_sold", "sum"),
            promotion_rate=("promotion_flag", "mean"),
            avg_discount=("discount_rate", "mean"),
            promotion_cost=("promotion_cost", "sum"),
        )
        .reset_index()
    )

    uplift = calculate_sales_uplift(df)

    category_uplift = (
        uplift.groupby("category")
        .agg(
            sales_uplift=("sales_uplift", "sum"),
            baseline_sales=("baseline_sales", "sum"),
        )
        .reset_index()
    )

    category_summary = category_summary.merge(
        category_uplift, on="category", how="left"
    )

    category_summary["sales_uplift"] = category_summary["sales_uplift"].fillna(0)
    category_summary["baseline_sales"] = category_summary["baseline_sales"].fillna(0)

    category_summary["uplift_rate"] = category_summary.apply(
        lambda row: row["sales_uplift"] / row["baseline_sales"]
        if row["baseline_sales"] > 0
        else 0,
        axis=1,
    )

    numeric_cols = [
        "total_sales",
        "promotion_rate",
        "avg_discount",
        "promotion_cost",
        "sales_uplift",
        "baseline_sales",
        "uplift_rate",
    ]

    for col in numeric_cols:
        category_summary[col] = category_summary[col].round(3)

    return category_summary.sort_values("sales_uplift", ascending=False)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect simple sales anomalies using z-score within each category.
    """

    result = df.copy()

    result["category_mean_sales"] = result.groupby("category")["sales"].transform("mean")
    result["category_std_sales"] = result.groupby("category")["sales"].transform("std")

    result["sales_z_score"] = (
        result["sales"] - result["category_mean_sales"]
    ) / result["category_std_sales"]

    anomalies = result[result["sales_z_score"].abs() >= 2.5].copy()

    columns = [
        "date",
        "category",
        "region",
        "channel",
        "campaign_name",
        "sales",
        "sales_z_score",
        "promotion_flag",
        "discount_rate",
    ]

    anomalies = anomalies[columns]
    anomalies["sales_z_score"] = anomalies["sales_z_score"].round(3)

    return anomalies.sort_values("sales_z_score", ascending=False)