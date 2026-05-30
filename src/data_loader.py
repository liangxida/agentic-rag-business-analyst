from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "category",
    "region",
    "channel",
    "sales",
    "units_sold",
    "promotion_flag",
    "discount_rate",
    "campaign_name",
    "promotion_cost",
]


def load_sales_data(file_path: str) -> pd.DataFrame:
    """
    Load sales data from CSV and perform basic type conversion.
    """

    df = pd.read_csv(file_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "sales",
        "units_sold",
        "promotion_flag",
        "discount_rate",
        "promotion_cost",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a data quality report as a DataFrame.
    """

    report = []

    report.append(
        {
            "check": "row_count",
            "result": len(df),
            "status": "info",
        }
    )

    duplicate_count = df.duplicated().sum()
    report.append(
        {
            "check": "duplicate_rows",
            "result": duplicate_count,
            "status": "pass" if duplicate_count == 0 else "warning",
        }
    )

    for col in REQUIRED_COLUMNS:
        missing_count = df[col].isna().sum()
        report.append(
            {
                "check": f"missing_values_in_{col}",
                "result": int(missing_count),
                "status": "pass" if missing_count == 0 else "warning",
            }
        )

    negative_sales = (df["sales"] < 0).sum()
    report.append(
        {
            "check": "negative_sales",
            "result": int(negative_sales),
            "status": "pass" if negative_sales == 0 else "warning",
        }
    )

    invalid_promotion_flag = (~df["promotion_flag"].isin([0, 1])).sum()
    report.append(
        {
            "check": "invalid_promotion_flag",
            "result": int(invalid_promotion_flag),
            "status": "pass" if invalid_promotion_flag == 0 else "warning",
        }
    )

    return pd.DataFrame(report)


def summarize_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate high-level summary by category.
    """

    summary = (
        df.groupby("category")
        .agg(
            total_sales=("sales", "sum"),
            total_units=("units_sold", "sum"),
            avg_discount=("discount_rate", "mean"),
            promotion_rate=("promotion_flag", "mean"),
            total_promotion_cost=("promotion_cost", "sum"),
        )
        .reset_index()
    )

    summary["total_sales"] = summary["total_sales"].round(2)
    summary["avg_discount"] = summary["avg_discount"].round(3)
    summary["promotion_rate"] = summary["promotion_rate"].round(3)
    summary["total_promotion_cost"] = summary["total_promotion_cost"].round(2)

    return summary