from __future__ import annotations

from src.etl_pipeline import ETLConfig, run_etl_pipeline
from src.sql_tools import create_sqlite_database
from src.agent_evaluator import AgentEvaluationConfig, run_agent_evaluation


def main() -> None:
    """
    Run the automated data pipeline.

    Steps:
    1. Extract raw sales data
    2. Validate schema and data quality
    3. Clean and transform KPI-ready features
    4. Export cleaned data and data quality report
    5. Load cleaned data into SQLite for downstream agent tools
    """

    config = ETLConfig(
        raw_csv_path="data/sample_sales.csv",
        cleaned_csv_path="data/cleaned_sales.csv",
        quality_report_path="outputs/data_quality_report.csv",
    )

    etl_result = run_etl_pipeline(config)

    create_sqlite_database(
        csv_path=etl_result["cleaned_csv_path"],
        db_path="data/business_analytics.db",
        table_name="sales",
    )

    print("Automated pipeline completed.")
    print(f"Raw data: {etl_result['raw_csv_path']}")
    print(f"Cleaned data: {etl_result['cleaned_csv_path']}")
    print(f"Data quality report: {etl_result['quality_report_path']}")
    print(f"Rows before cleaning: {etl_result['rows_before_cleaning']}")
    print(f"Rows after cleaning: {etl_result['rows_after_cleaning']}")
    print("SQLite database refreshed: data/business_analytics.db")

    evaluation_config = AgentEvaluationConfig(
        data_path=etl_result["cleaned_csv_path"],
        context_path="data/business_context.md",
        output_path="outputs/agent_evaluation_report.csv",
    )

    evaluation_report = run_agent_evaluation(evaluation_config)

    print(f"Agent evaluation report: {evaluation_config.output_path}")
    print(f"Evaluated questions: {len(evaluation_report)}")

if __name__ == "__main__":
    main()