from __future__ import annotations

import os
from dataclasses import dataclass

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


@dataclass
class ETLConfig:
    raw_csv_path: str = "data/sample_sales.csv"
    cleaned_csv_path: str = "data/cleaned_sales.csv"
    quality_report_path: str = "outputs/data_quality_report.csv"


def extract_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Extract raw sales data from CSV.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate whether the raw data contains all required columns.
    """

    rows = []

    existing_columns = set(df.columns)

    for col in REQUIRED_COLUMNS:
        status = "pass" if col in existing_columns else "fail"
        rows.append(
            {
                "check_type": "schema",
                "check_name": f"required_column_{col}",
                "result": col in existing_columns,
                "status": status,
                "details": "" if status == "pass" else f"Missing required column: {col}",
            }
        )

    extra_columns = sorted(list(existing_columns - set(REQUIRED_COLUMNS)))
    rows.append(
        {
            "check_type": "schema",
            "check_name": "extra_columns",
            "result": len(extra_columns),
            "status": "info",
            "details": ", ".join(extra_columns) if extra_columns else "No extra columns",
        }
    )

    return pd.DataFrame(rows)


def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns into expected data types.
    """

    result = df.copy()

    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    numeric_columns = [
        "sales",
        "units_sold",
        "promotion_flag",
        "discount_rate",
        "promotion_cost",
    ]

    for col in numeric_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    string_columns = [
        "product_id",
        "category",
        "region",
        "channel",
        "campaign_name",
    ]

    for col in string_columns:
        result[col] = result[col].astype(str).str.strip()

    return result


def run_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run data quality checks and return a report.
    """

    rows = []

    rows.append(
        {
            "check_type": "completeness",
            "check_name": "row_count",
            "result": len(df),
            "status": "info",
            "details": "Total number of rows in the dataset",
        }
    )

    duplicate_count = int(df.duplicated().sum())
    rows.append(
        {
            "check_type": "uniqueness",
            "check_name": "duplicate_rows",
            "result": duplicate_count,
            "status": "pass" if duplicate_count == 0 else "warning",
            "details": "Duplicate rows detected" if duplicate_count > 0 else "No duplicate rows",
        }
    )

    for col in REQUIRED_COLUMNS:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "check_type": "completeness",
                "check_name": f"missing_values_{col}",
                "result": missing_count,
                "status": "pass" if missing_count == 0 else "warning",
                "details": f"{missing_count} missing values in {col}",
            }
        )

    negative_sales_count = int((df["sales"] < 0).sum())
    rows.append(
        {
            "check_type": "validity",
            "check_name": "negative_sales",
            "result": negative_sales_count,
            "status": "pass" if negative_sales_count == 0 else "warning",
            "details": "Sales should not be negative",
        }
    )

    negative_units_count = int((df["units_sold"] < 0).sum())
    rows.append(
        {
            "check_type": "validity",
            "check_name": "negative_units_sold",
            "result": negative_units_count,
            "status": "pass" if negative_units_count == 0 else "warning",
            "details": "Units sold should not be negative",
        }
    )

    invalid_promo_flag_count = int((~df["promotion_flag"].isin([0, 1])).sum())
    rows.append(
        {
            "check_type": "validity",
            "check_name": "invalid_promotion_flag",
            "result": invalid_promo_flag_count,
            "status": "pass" if invalid_promo_flag_count == 0 else "warning",
            "details": "promotion_flag should be 0 or 1",
        }
    )

    invalid_discount_count = int(
        ((df["discount_rate"] < 0) | (df["discount_rate"] > 1)).sum()
    )
    rows.append(
        {
            "check_type": "validity",
            "check_name": "invalid_discount_rate",
            "result": invalid_discount_count,
            "status": "pass" if invalid_discount_count == 0 else "warning",
            "details": "discount_rate should be between 0 and 1",
        }
    )

    sales_mean = df["sales"].mean()
    sales_std = df["sales"].std()

    if pd.isna(sales_std) or sales_std == 0:
        outlier_count = 0
    else:
        z_score = (df["sales"] - sales_mean) / sales_std
        outlier_count = int((z_score.abs() >= 3).sum())

    rows.append(
        {
            "check_type": "outlier",
            "check_name": "sales_outliers_z_score_3",
            "result": outlier_count,
            "status": "pass" if outlier_count == 0 else "warning",
            "details": "Rows with absolute sales z-score >= 3",
        }
    )

    return pd.DataFrame(rows)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sales data for downstream analytics and agent tools.
    """

    result = df.copy()

    result = result.drop_duplicates()

    result = result.dropna(subset=["date", "category", "region", "sales", "units_sold"])

    result = result[result["sales"] >= 0]
    result = result[result["units_sold"] >= 0]
    result = result[result["promotion_flag"].isin([0, 1])]
    result = result[(result["discount_rate"] >= 0) & (result["discount_rate"] <= 1)]

    result["campaign_name"] = result["campaign_name"].fillna("No Campaign")
    result["promotion_cost"] = result["promotion_cost"].fillna(0)

    return result


def transform_kpi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create KPI-ready features for downstream analysis.
    """

    result = df.copy()

    result["month"] = result["date"].dt.to_period("M").astype(str)
    result["week"] = result["date"].dt.to_period("W").astype(str)

    result["revenue_per_unit"] = result.apply(
        lambda row: row["sales"] / row["units_sold"] if row["units_sold"] > 0 else 0,
        axis=1,
    )

    result["is_discounted"] = (result["discount_rate"] > 0).astype(int)

    result["promotion_intensity"] = result["promotion_flag"] * result["discount_rate"]

    result["sales_per_promotion_dollar"] = result.apply(
        lambda row: row["sales"] / row["promotion_cost"]
        if row["promotion_cost"] > 0
        else 0,
        axis=1,
    )

    return result


def save_outputs(
    cleaned_df: pd.DataFrame,
    quality_report: pd.DataFrame,
    cleaned_csv_path: str,
    quality_report_path: str,
) -> None:
    """
    Save cleaned data and quality report.
    """

    os.makedirs(os.path.dirname(cleaned_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(quality_report_path), exist_ok=True)

    cleaned_df.to_csv(cleaned_csv_path, index=False)
    quality_report.to_csv(quality_report_path, index=False)


def run_etl_pipeline(config: ETLConfig | None = None) -> dict:
    """
    Run the full ETL pipeline.

    Returns paths and basic metadata for downstream automation.
    """

    if config is None:
        config = ETLConfig()

    raw_df = extract_raw_data(config.raw_csv_path)

    schema_report = validate_schema(raw_df)

    failed_schema_checks = schema_report[schema_report["status"] == "fail"]
    if not failed_schema_checks.empty:
        raise ValueError(
            "Schema validation failed. Missing required columns: "
            + ", ".join(failed_schema_checks["check_name"].tolist())
        )

    typed_df = standardize_data_types(raw_df)
    quality_report = run_data_quality_checks(typed_df)
    cleaned_df = clean_sales_data(typed_df)
    transformed_df = transform_kpi_features(cleaned_df)

    final_quality_report = pd.concat(
        [
            schema_report,
            quality_report,
            pd.DataFrame(
                [
                    {
                        "check_type": "etl_summary",
                        "check_name": "rows_before_cleaning",
                        "result": len(raw_df),
                        "status": "info",
                        "details": "Number of rows before cleaning",
                    },
                    {
                        "check_type": "etl_summary",
                        "check_name": "rows_after_cleaning",
                        "result": len(transformed_df),
                        "status": "info",
                        "details": "Number of rows after cleaning and transformation",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    save_outputs(
        cleaned_df=transformed_df,
        quality_report=final_quality_report,
        cleaned_csv_path=config.cleaned_csv_path,
        quality_report_path=config.quality_report_path,
    )

    return {
        "raw_csv_path": config.raw_csv_path,
        "cleaned_csv_path": config.cleaned_csv_path,
        "quality_report_path": config.quality_report_path,
        "rows_before_cleaning": len(raw_df),
        "rows_after_cleaning": len(transformed_df),
    }


if __name__ == "__main__":
    result = run_etl_pipeline()
    print("ETL pipeline completed.")
    print(result)