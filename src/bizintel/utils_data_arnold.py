"""utils_data.py - reusable data loading and inspection functions.

These functions work on any pandas DataFrame.
"""

# === IMPORTS ===

from pathlib import Path

import pandas as pd

from bizintel.utils_logger import LOG

# === PANDAS DISPLAY CONFIGURATION ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === FUNCTIONS ===


# Function to load data from a CSV file
def load_data(filepath: Path, name: str) -> pd.DataFrame:
    """Load one CSV file into a pandas DataFrame.

    WHY: Reading from CSV into a DataFrame is the first step
    in almost every BI pipeline.


    Args:
        filepath: Path to the CSV file.
        name: A short name for logging.

    Returns:
        A pandas DataFrame with the file contents.
    """
    LOG.info(f"Loading {name} from {filepath}")
    df: pd.DataFrame = pd.read_csv(filepath)
    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# Function to inspect basic structure of a DataFrame
def inspect_basic(df: pd.DataFrame, name: str) -> None:
    """Log basic structure of a DataFrame.

    WHY: Before analysis, we need to understand what columns
    exist, what types they are, and what the first few rows look like.

    Args:
        df: The DataFrame to inspect.
        name: A short name for logging.

    Returns:
        None
    """
    LOG.info(f"Inspecting {name}")
    LOG.info(f"  Columns: {list(df.columns)}")
    LOG.info(f"  Shape:   {df.shape[0]} rows x {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}\n")


# Function to check data quality (missing values, duplicates, invalid words, outliers)
def check_quality(df: pd.DataFrame, name: str) -> None:
    """Check for missing values and duplicate rows.

    WHY: Real-world data is messy. Knowing what problems exist
    helps us decide how to handle them in the cleaning step.

    Args:
        df: The DataFrame to check.
        name: A short name for logging.

    Returns:
        None
    """
    LOG.info(f"Quality check: {name}")

    LOG.info(f"Checking for missing values in {name}")

    missing: pd.Series = df.isna().sum()
    total_missing: int = int(missing.sum())
    LOG.info(f"  Total missing values: {total_missing}")

    if total_missing > 0:
        LOG.warning(f"  Missing by column:\n{missing[missing > 0]}")

    LOG.info(
        f"Checking for words like 'unknown' or 'error' or 'other' or 'undefined'in {name}"
    )
    invalid_words = ["unknown", "error", "other", "undefined"]
    for col in df.columns:
        for word in invalid_words:
            count = (
                df[col]
                .astype(str)
                .str.contains(rf"\b{word}\b", case=False, na=False)
                .sum()
            )
            if count > 0:
                LOG.warning(f"  Column '{col}' has {count} occurrences of '{word}'")
            else:
                LOG.debug(f"  Column '{col}' has no occurrences of '{word}'")

    # Checking for duplicate rows
    LOG.info(f"Checking for duplicate rows in {name}")
    duplicate_count: int = int(df.duplicated().sum())
    LOG.info(f"  Duplicate rows: {duplicate_count}")

    if duplicate_count > 0:
        LOG.warning(
            f"  {duplicate_count} duplicate row(s) found - "
            "review before loading to warehouse"
        )
    # view duplicate rows
    if duplicate_count > 0:
        LOG.debug(f"  Duplicate rows:\n{df[df.duplicated(keep=False)]}")

    # Check for outliers in numeric columns using the IQR method.
    LOG.info(f"Outlier check: {name}")

    numeric_cols = df.select_dtypes(include="number")
    if numeric_cols.empty:
        LOG.info(f"  No numeric columns in {name}")
        return

    for col in numeric_cols.columns:
        Q1 = numeric_cols[col].quantile(0.25)
        Q3 = numeric_cols[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = numeric_cols[
            (numeric_cols[col] < Q1 - 1.5 * IQR) | (numeric_cols[col] > Q3 + 1.5 * IQR)
        ]
        outlier_count = outliers.shape[0]
        LOG.info(f"  Column '{col}' has {outlier_count} outlier(s)")
        if outlier_count > 0:
            LOG.debug(f"  Outliers in column '{col}':\n{outliers}")


# Function to summarize numeric columns in a DataFrame.
def summarize_numeric(df: pd.DataFrame, name: str) -> None:
    """Log basic descriptive statistics for numeric columns.

    WHY: Summary statistics give a quick sense of the range
    and distribution of numeric data.
    Unexpected values often signal data quality issues.

    Args:
        df: The DataFrame to summarize.
        name: A short name for logging.

    Returns:
        None
    """
    numeric_cols = df.select_dtypes(include="number")
    if numeric_cols.empty:
        LOG.info(f"  No numeric columns in {name}")
        return

    LOG.info(f"Numeric summary: {name}")
    LOG.info(f"\n{numeric_cols.describe().round(2)}")


# Function to summarize categorical columns in a DataFrame.
def summarize_categorical(df: pd.DataFrame, name: str) -> None:
    """Log basic descriptive statistics for categorical columns.

    WHY: Summary statistics give a quick sense of the distribution
    of categorical data. Unexpected values often signal data quality issues.

    Args:
        df: The DataFrame to summarize.
        name: A short name for logging.

    Returns:
        None
    """
    categorical_cols = df.select_dtypes(include="object")
    if categorical_cols.empty:
        LOG.info(f"  No categorical columns in {name}")
        return

    LOG.info(f"Categorical summary: {name}")
    for col in categorical_cols.columns:
        value_counts = categorical_cols[col].value_counts(dropna=False)
        LOG.info(f"  Column '{col}':\n{value_counts}")


# Define function to prep a DataFrame for downstream use.
def prep_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Prepare a DataFrame for downstream use by handling missing values and duplicates.

    WHY: Cleaning and prepping the DataFrame ensures consistent and reliable analysis.

    Args:
        df: The DataFrame to prep.
        name: A short name for logging.

    Returns:
        The cleaned DataFrame.
    """
    LOG.info(f"Prepping DataFrame: {name}")

    # Format column names to be consistent.Capitalize each word and remove leading/trailing/middle spaces
    LOG.info(f"Formatting column names for DataFrame: {name}")
    df.columns = df.columns.str.strip().str.title().str.replace(" ", "")
    LOG.info(f"  Formatted column names: {df.columns.tolist()}")

    # Drop duplicate rows
    LOG.info(f"Removing duplicate rows for DataFrame: {name}")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        LOG.info(f"  Dropping {duplicate_count} duplicate row(s)")
        df = df.drop_duplicates()
    else:
        LOG.info(f"  No duplicate rows found for DataFrame: {name}")

    # Convert invalid words to blank values (handles all columns, case, and spacing variations)
    invalid_words = ["unknown", "error", "other", "undefined"]

    for col in df.columns:
        df[col] = (
            df[col]
            .astype("string")  # temporarily convert all columns to string
            .str.strip()
            .str.lower()
            .replace(invalid_words, " ")
        )
    LOG.info(f"Converted invalid words to blank values for DataFrame: {name}")

    # Convert columns to appropriate data types if necessary
    LOG.info(f"Converting columns to appropriate data types for DataFrame: {name}")
    for col in df.columns:
        if col.lower().endswith("date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            LOG.info(f"  Converted column '{col}' to datetime")
        elif (
            col.lower().endswith("id")
            or col.lower().endswith("location")
            or col.lower().endswith("method")
            or col.lower().endswith("item")
        ):
            df[col] = df[col].astype(str)
            LOG.info(f"  Converted column '{col}' to string")
        elif (
            col.lower().endswith("quantity")
            or col.lower().endswith("unit")
            or col.lower().endswith("spent")
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            LOG.info(f"  Converted column '{col}' to numeric")

    # Check dtypes
    LOG.info(f"Data types for DataFrame: {name}")
    for col in df.columns:
        LOG.info(f"  Column '{col}' has dtype: {df[col].dtype}")

    # Dealing with missing values in Quantity, PricePerUnit, and TotalSpent columns
    # Median values by Item
    price_by_item = df.groupby("Item")["PricePerUnit"].median()
    quantity_by_item = df.groupby("Item")["Quantity"].median()

    # Median values overall, used when there is no specific item median available
    overall_price = df["PricePerUnit"].median()
    overall_quantity = df["Quantity"].median()

    for idx, row in df.iterrows():
        q = row["Quantity"]
        p = row["PricePerUnit"]
        t = row["TotalSpent"]

        missing = pd.isna([q, p, t]).sum()

        # ------------------------------------------
        # One value missing: calculate directly
        # ------------------------------------------
        if missing == 1:
            if pd.isna(q) and pd.notna(t) and pd.notna(p) and p != 0:
                df.at[idx, "Quantity"] = int(round(t / p))

            elif pd.isna(p) and pd.notna(t) and pd.notna(q) and q != 0:
                df.at[idx, "PricePerUnit"] = t / q

            elif pd.isna(t) and pd.notna(q) and pd.notna(p):
                df.at[idx, "TotalSpent"] = q * p

        # ------------------------------------------
        # Two values missing
        # ------------------------------------------
        elif missing == 2:
            # Only TotalSpent exists
            if pd.notna(t):
                price = price_by_item.get(row["Item"], overall_price)

                if pd.notna(price) and price != 0:
                    df.at[idx, "PricePerUnit"] = price
                    df.at[idx, "Quantity"] = int(round(t / price))

            # Only PricePerUnit exists
            elif pd.notna(p):
                quantity = quantity_by_item.get(row["Item"], overall_quantity)

                if pd.notna(quantity):
                    df.at[idx, "Quantity"] = quantity
                    df.at[idx, "TotalSpent"] = quantity * p

            # Only Quantity exists
            elif pd.notna(q):
                price = price_by_item.get(row["Item"], overall_price)

                if pd.notna(price):
                    df.at[idx, "PricePerUnit"] = price
                    df.at[idx, "TotalSpent"] = q * price

        # ------------------------------------------
        # All three missing
        # ------------------------------------------
        elif missing == 3:
            quantity = quantity_by_item.get(row["Item"], overall_quantity)
            price = price_by_item.get(row["Item"], overall_price)

            if pd.notna(quantity) and pd.notna(price):
                df.at[idx, "Quantity"] = quantity
                df.at[idx, "PricePerUnit"] = price
                df.at[idx, "TotalSpent"] = quantity * price

    # Fill missing values for categorical columns with the mode
    categorical_cols = df.select_dtypes(include="object")
    for col in categorical_cols.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            mode_value = df[col].mode()[0]
            LOG.info(
                f"  Filling {missing_count} missing value(s) in column '{col}' with mode: {mode_value}"
            )
            df[col] = df[col].fillna(mode_value)

    # Remove any remaining rows with missing values
    df = df.dropna()
    LOG.info(f"  Removed rows with missing values, remaining rows: {len(df)}")

    # Remove leading and trailing whitespace from all string columns
    for col in categorical_cols.columns:
        df[col] = df[col].str.strip()
    return df
