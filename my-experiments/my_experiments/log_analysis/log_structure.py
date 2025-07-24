import pandas as pd
import argparse
import sys
import traceback


def analyze_log_structure(log_file_path: str):
    """
    Loads a log file into a pandas DataFrame and analyzes its structure by
    counting the occurrences of unique values in each column.

    Args:
        log_file_path (str): The path to the log file (e.g., .bz2, .csv).
    """
    try:
        print(f"--- Loading log file from: {log_file_path} ---")

        # Determine the file type and load accordingly.
        if log_file_path.endswith('.bz2'):
            df = pd.read_pickle(log_file_path, compression='bz2')
        elif log_file_path.endswith('.csv'):
            df = pd.read_csv(log_file_path)
        else:
            print(f"Error: Unsupported file type for {log_file_path}. Please use .bz2 or .csv.", file=sys.stderr)
            return False

        print("Log file loaded successfully.")

        if df.empty:
            print("\n--- The log file is empty. No data to analyze. ---")
            return True

        # --- Analyze each column ---
        print("\n--- Value Counts for Each Column ---")

        for column in df.columns:
            print(f"\n\n--- Analysis for Column: '{column}' ---")

            try:
                # Use value_counts() to get the frequency of each unique item.
                # dropna=False includes counts of NaN values.
                value_counts = df[column].value_counts(dropna=False)

                print(f"  - Data Type: {df[column].dtype}")
                print(f"  - Number of Unique Values: {len(value_counts)}")
                print("  - Value Counts (Top 20):")

                # Print the top 20 most frequent values for brevity
                print(value_counts.head(20).to_string())

            except TypeError:
                # This can happen if a column contains unhashable types like lists or dicts
                print(
                    f"  - Note: Cannot perform value_counts on this column because it contains complex objects (e.g., lists, dicts).")
                print(f"  - Data Type: {df[column].dtype}")
                print(f"  - First 5 entries: \n{df[column].head().to_string()}")


    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {log_file_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze the structure of a log file by counting unique values in each column."
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the log file (.bz2 or .csv)."
    )
    args = parser.parse_args()

    if not analyze_log_structure(args.log_file):
        sys.exit(1)
