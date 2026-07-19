"""data_prep_cafe_arnold.py - example.

An example of cleaning and preparing cafe_sales data.
Cleaning and preparation is a critical step in any BI workflow.
It is different for every project and every dataset.

Cleaning can be 80-90% of the work in a BI project.
It is often the most time-consuming step and
to do it well requires domain knowledge, attention to detail,
tenacity, and resourcefulness.

It is often the most important step because
if the data is not clean, the analysis will be wrong and
the business decisions will be wrong.

Common cleaning and preparation tasks include:
- Remove duplicate rows.
- Remove rows with missing or invalid values.
- Normalize inconsistent values (e.g., "East", "east", " EAST ").
- Convert data types (e.g., text to numeric, text to datetime).

Author: Arnold Atchoe.
Date: 2026-07.

Process:
    - Load raw CSV data files.
    -Explore raw data to understand its structure and quality.
    - Clean and prepare dataset.
    - Verify data quality after cleaning.
    - Save prepared data to data/prepared/.

Data Source:
- data/raw/dirty_cafe_sales.csv

Output:
- data/prepared/dirty_cafe_sales_data_prepared.csv

Terminal command to run this file from the root project folder:

uv run python -m bizintel.data_prep_cafe_arnold

OBS:
  Don't edit this file - it should remain a working example.
  Copy it, rename it with your alias, and modify your copy.
  If you do, include your command to run it in the docstring above and in README.md.
"""
# NOTE:We making use of resuable functions from utils_data.py for loading,
# inspecting, and checking data quality.
# === DECLARE IMPORTS (bring in free code from elsewhere) ===

from pathlib import Path
from typing import Final

from datafun_toolkit.logger import log_path
import pandas as pd

from bizintel.utils_data_arnold import (
    check_quality,
    inspect_basic,
    load_data,
    prep_dataframe,
    summarize_numeric,
)
from bizintel.utils_logger import LOG, log_header

# === DECLARE GLOBAL CONSTANTS AND CONFIGURATION ===

# Raw data folder path (relative to the root project folder).
DATA_RAW: Final[Path] = Path("data/raw")

# Prepared data folder path (relative to the root project folder).
DATA_PREPARED: Final[Path] = Path("data/prepared")

# Input files.
DIRTY_CAFE_SALES_FILE: Final[Path] = DATA_RAW / "dirty_cafe_sales.csv"

# Output files.
CAFE_SALES_PREPARED: Final[Path] = DATA_PREPARED / "dirty_cafe_sales_data_prepared.csv"


# === NOTE: All Functions are found in utils_data_arnold.py ===


def save_prepared(df: pd.DataFrame, filepath: Path, name: str) -> None:
    """Save a prepared DataFrame to CSV.

    WHY: Saving prepared data to a separate folder keeps raw data
    untouched and gives downstream steps a clean input to work from.

    Args:
        df: Prepared DataFrame to save.
        filepath: Path to the output CSV file.
        name: A short name for logging.

    Returns:
        None
    """
    # Create the output folder if it does not exist
    # parents=True means create any missing parent folders as well.
    # exist_ok=True means
    # do not raise an error (it's ok) if the folder already exists.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Call the df.to_csv() method to save the DataFrame to a CSV file.
    # Pass in the filepath where the file should be saved.
    # Set the index parameter to False to avoid saving the index column.
    # This is important because the index is not part of the original data
    # and should not be included in the saved file.
    df.to_csv(filepath, index=False)

    # Use df.shape[0] to get the number of rows in the DataFrame.
    row_count: int = df.shape[0]

    # Log useful information about the saved file,
    # including the number of rows and the file path.
    LOG.info(f"Saved {name}")
    LOG.info(f"  Rows: {row_count}")
    LOG.info(f"  Path: {filepath}")


# === MAIN FUNCTION ===


def main() -> None:
    """Main function to run the data preparation logic.

    This is where the main logic starts
    when this script is run.
    """
    # First, log the header for the BI module to indicate the start of the workflow.
    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "Raw data:     ", DATA_RAW)
    log_path(LOG, "Prepared data:", DATA_PREPARED)

    LOG.info("Task 1. LOAD. Call a function to load each dataset......")
    df_cafe_sales = load_data(DIRTY_CAFE_SALES_FILE, "cafe sales")

    LOG.info("Task 2. INSPECT. Call a function to inspect each dataset...")
    inspect_basic(df_cafe_sales, "cafe sales")

    LOG.info("Task 3. CHECK QUALITY BEFORE........")
    check_quality(df_cafe_sales, "cafe sales")

    LOG.info("Task 4. SUMMARIZE BEFORE.......... ")
    summarize_numeric(df_cafe_sales, "cafe sales")

    LOG.info("Task 5. PREPARE DATASETS.........")
    df_cafe_sales_prepared = prep_dataframe(df_cafe_sales, "cafe sales")

    # For cafe sales, we don't have separate customer and product datasets.
    # So we directly use the prepared cafe sales DataFrame.
    df_cafe_sales_prepared = df_cafe_sales_prepared

    LOG.info("Task 6. CHECK QUALITY AFTER PREPARATION........")
    check_quality(df_cafe_sales_prepared, "cafe sales prepared")

    LOG.info("Task 7. SUMMARIZE AFTER PREPARATION........")
    summarize_numeric(df_cafe_sales_prepared, "cafe sales prepared")

    LOG.info("Task 8. SAVE PREPARED DATASETS........")
    save_prepared(df_cafe_sales_prepared, CAFE_SALES_PREPARED, "cafe sales")

    LOG.info("Workflow complete")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    # This conditional ensures that main() is only called
    # when this script is run directly, not when imported.
    main()
