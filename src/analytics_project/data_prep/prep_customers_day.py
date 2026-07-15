"""The purpose of this script is to ensure that our customers table
is clean and ready for down stream analysis. This script will perform the following tasks:
1. Remove any duplicate records from the customers table.
2. Address any missing values in the customers table.
3. Ensure that the data types of each column are appropriate for analysis.
4. Address any issues related with outliers in the customers table.
5. Standardize the format of the data in the customers table.
"""

### Import necessary libraries

# Import standard libraries
import pathlib
import sys

# Import external libraries
import pandas as pd

# Import utils_logger for logging
from bizintel.utils_logger import LOG

# Ensure that the current working directory is set to the root of the project
working_dir = pathlib.Path(__file__).resolve()
module_dir = working_dir.parent.parent.parent
sys.path.append(
    str(module_dir)
)  # This allows us to import modules from the root of the project


# Constants
CLEANED_DATA_PATH = (
    module_dir.parent / "data" / "processed"
)  # Where cleaned data will be stored.
RAW_DATA_PATH = module_dir.parent / "data" / "raw"  # Where raw data is stored.

# Check if directories exist, if not create them
CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
LOG.info(f"Ensured that the directory {CLEANED_DATA_PATH} exists.")
LOG.info(f"Ensured that the directory {RAW_DATA_PATH} exists.")

### Define functions for data cleaning.


# Define a function to fetch raw data from a specified file
def fetch_raw_data(file_name: str) -> pd.DataFrame:
    """Fetches the raw data from the specified file.

    Args:
        file_name (str): The name of the file to fetch"""
    file_path: pathlib.Path = RAW_DATA_PATH.joinpath(file_name)
    try:
        LOG.info(f"Attempting to read the raw data from {file_path}.")
        data = pd.read_csv(file_path)
        LOG.info(f"Successfully read the raw data from {file_path}.")
        return data
    except FileNotFoundError:
        LOG.error(
            f"File {file_path} not found. Please check the file path and try again."
        )
        raise
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    except Exception as e:
        LOG.error(f"An error occurred while reading the file {file_path}: {e}")
        raise
        return pd.DataFrame()  # Return an empty DataFrame in case of error


# Define a function to inspect data.
def inspect_data(df: pd.DataFrame) -> None:
    """Inspects the data for basic information.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    LOG.info("Inspecting the data for basic information.")
    LOG.info(f"DataFrame shape: {df.shape}")
    LOG.info(f"DataFrame columns: {df.columns.tolist()}")
    LOG.info(f"DataFrame info:\n{df.info()}")
    LOG.info(f"DataFrame head:\n{df.head()}")
    LOG.info(f"DataFrame description:\n{df.describe(include='number')}")


# Define function identify and delete rows with missing values.
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and deletes rows with missing values.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: The DataFrame after handling missing values.
    """
    LOG.info("Identifying and deleting rows with missing values.")
    initial_shape = df.shape
    missing_values = df.isnull().sum()
    # if missing is less than 5% of the total rows, we can drop those rows.
    # Otherwise populate the missing values for numeric columns with the mean and for categorical columns with the mode.
    if (missing_values.sum() / df.shape[0]) < 0.05:
        LOG.info(
            "Missing values are less than 5% of total rows. Dropping rows with missing values."
        )
        df_cleaned = df.dropna()
    else:
        LOG.info(
            "Missing values are more than 5% of total rows. Imputing missing values."
        )
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                LOG.info(
                    f"Imputed missing values in numeric column '{column}' with mean value {mean_value}."
                )
            else:
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
                LOG.info(
                    f"Imputed missing values in categorical column '{column}' with mode value '{mode_value}'."
                )
        df_cleaned = df
    final_shape = df_cleaned.shape
    LOG.info(f"Initial DataFrame shape: {initial_shape}")
    LOG.info(f"Missing values per column:\n{missing_values}")
    LOG.info(f"Final DataFrame shape after handling missing values: {final_shape}")
    return df_cleaned


# Define function to identify and delete duplicate rows.
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and deletes duplicate rows.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    Returns:
        pd.DataFrame: The DataFrame after handling duplicates.
    """
    LOG.info("Identifying and deleting duplicate rows.")
    initial_shape = df.shape
    duplicate_count = df.duplicated().sum()
    LOG.info(f"Found {duplicate_count} duplicate rows.")
    df_cleaned = df.drop_duplicates()
    final_shape = df_cleaned.shape
    LOG.info(f"Initial DataFrame shape: {initial_shape}")
    LOG.info(f"Final DataFrame shape after handling duplicates: {final_shape}")
    return df_cleaned


# Define function to deal with outliers in the data.
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and handles outliers in the data.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: The DataFrame after handling outliers.
    """
    LOG.info("Identifying and handling outliers in the data.")
    initial_shape = df.shape
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count = outliers.shape[0]
        LOG.info(f"Found {outlier_count} outliers in column '{column}'.")
        # Remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    final_shape = df.shape
    LOG.info(f"Initial DataFrame shape: {initial_shape}")
    LOG.info(f"Final DataFrame shape after handling outliers: {final_shape}")
    return df


# Define function to standardize data types and formats.
def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    LOG.info("Ensuring that certain columns are formatted correctly")
    if "name" in df.columns:
        name_parts = df["name"].astype(str).str.split(" ", n=1, expand=True)
        df["firstname"] = name_parts[0]
        df["lastname"] = name_parts[1].fillna("")
        LOG.info("Split 'Name' column into 'FirstName' and 'LastName'.")
        df = df.drop(columns=["name"])
    if "joinDate" in df.columns:
        df["joinDate"] = pd.to_datetime(df["joinDate"], errors="coerce")
        LOG.info("Converted 'joinDate' to datetime.")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower()
    return df


# Define function to save prepared data to a folder
def save_prepared_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save cleaned data to CSV.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        file_name (str): Name of the output file.
    """
    LOG.info(
        f"FUNCTION START: save_prepared_data with file_name={file_name}, dataframe shape={df.shape}"
    )
    file_path = CLEANED_DATA_PATH.joinpath(file_name)
    df.to_csv(file_path, index=False)
    LOG.info(f"Data saved to {file_path}")


# Define Main function to clean customer data.
def main() -> None:
    LOG.info("==================================")
    LOG.info("STARTING prep_customers_day.py")
    LOG.info("==================================")

    input_file = "customers_data.csv"
    output_file = "customers_prepared.csv"

    df = fetch_raw_data(input_file)

    original_shape = df.shape
    LOG.info(f"Initial dataframe columns: {', '.join(df.columns.tolist())}")
    LOG.info(f"Initial dataframe shape: {df.shape}")

    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)

    changed_columns = [
        f"{old} -> {new}"
        for old, new in zip(original_columns, df.columns, strict=True)
        if old != new
    ]
    if changed_columns:
        LOG.info(f"Cleaned column names: {', '.join(changed_columns)}")

    df = handle_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = standardize_data(df)

    save_prepared_data(df, output_file)

    LOG.info("==================================")
    LOG.info(f"Original shape: {original_shape}")
    LOG.info(f"Cleaned shape:  {df.shape}")
    LOG.info("==================================")
    LOG.info("FINISHED prep_customers_day.py")
    LOG.info("==================================")


# -------------------
# Conditional Execution Block
# -------------------

if __name__ == "__main__":
    main()
