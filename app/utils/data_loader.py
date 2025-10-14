import os
import pandas as pd


def load_dataframe(file_path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    else:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            return pd.read_csv(file_path, nrows=nrows)
        elif ext == ".json":
            return pd.read_json(file_path)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(file_path)
        elif ext == ".parquet":
            return pd.read_parquet(file_path)
        # elif ext == ".zip":
        #     ... (extract then load)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
