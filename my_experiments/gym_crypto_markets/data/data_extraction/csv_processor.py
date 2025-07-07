import pandas as pd
import argparse
import ast
from tqdm import tqdm
import os
import sys
import zipfile

def extract_rename_and_save_columns(
    input_file_path,
    output_file_path,
    column_indices,
    new_column_names,
    include_output_header=True,
    include_output_index=False,
    buy_sell_flag_col_name=None,
    price_to_cents_col_name=None,
    timestamp_col_name=None,
    symbol=None
):
    """
    Extracts specific columns from an input CSV file, gives them new titles,
    converts a nanosecond-epoch timestamp column to pandas Timestamps,
    converts a boolean 'isBuyerMaker' column to "BUY"/"SELL" strings,
    multiplies a specified price column by 100, adds a symbol column,
    and saves the result to a new output CSV file. Includes progress indication.

    Args:
        input_file_path (str): The path to your input CSV file.
        output_file_path (str): The path where the new CSV file will be saved.
        column_indices (list): A list of 0-based integer indices of the columns to extract.
        new_column_names (list): A list of strings for the new titles.
        include_output_header (bool): True to include new column names as header in output.
        include_output_index (bool): True to include DataFrame index in output.
        buy_sell_flag_col_name (str, optional): The new column name for buy/sell flag conversion.
        price_to_cents_col_name (str, optional): The new column name for price to cents conversion.
        timestamp_col_name (str, optional): The new column name that holds the nanosecond
                                            timestamp to be converted to pandas Timestamp.
        symbol (str, optional): The symbol string (e.g., 'BTCUSDT') to add as a new column.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        print(f"[{pd.Timestamp.now()}] Starting processing...")

        file_size_bytes = 0
        try:
            file_size_bytes = os.path.getsize(input_file_path)
        except OSError:
            print(f"[{pd.Timestamp.now()}] Warning: Could not get size of '{input_file_path}' for reading progress bar.")

        print(f"[{pd.Timestamp.now()}] Reading input CSV '{input_file_path}'...")

        df_list = []
        df_check = pd.read_csv(input_file_path, nrows=5)
        print(df_check)
        print(column_indices)
        csv_reader_chunks = pd.read_csv(
            input_file_path,
            usecols=column_indices,
            header= None,
            index_col=False,
            chunksize=100000
        )


        with tqdm(total=file_size_bytes, unit='B', unit_scale=True, desc="Reading CSV") as pbar:
            for i, chunk in enumerate(csv_reader_chunks):
                processed_chunk = chunk[column_indices]

                processed_chunk.columns = new_column_names

                if len(processed_chunk.columns) != len(new_column_names):
                    raise ValueError(
                        f"Mismatched column counts! Extracted {len(processed_chunk.columns)} columns "
                        f"but received {len(new_column_names)} new names. "
                        f"Check your `column_indices` and `new_column_names`."
                    )

                df_list.append(processed_chunk)

            pbar.update(len(processed_chunk.to_csv(index=False).encode('utf-8')))

        df = pd.concat(df_list, ignore_index=True)
        print(df.head())

        print(f"[{pd.Timestamp.now()}] Finished reading. DataFrame loaded with {len(df):,} rows.")


        # Convert Microseconds Timestamp to pandas Timestamp
        if timestamp_col_name and timestamp_col_name in df.columns:
            print(f"[{pd.Timestamp.now()}] Converting '{timestamp_col_name}' to pandas Timestamp (unit='us')...")
            df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name], unit='us', errors='coerce')
            print(f"[{pd.Timestamp.now()}] Timestamp conversion complete.")

        # Convert Price to Cents from Dollars
        if price_to_cents_col_name and price_to_cents_col_name in df.columns:
            print(f"[{pd.Timestamp.now()}] Converting '{price_to_cents_col_name}' to cents (multiplying by 100)...")
            df[price_to_cents_col_name] = pd.to_numeric(df[price_to_cents_col_name], errors='coerce')
            df[price_to_cents_col_name] *= 100
            df[price_to_cents_col_name] = df[price_to_cents_col_name].astype(int)
            print(f"[{pd.Timestamp.now()}] Price conversion complete.")

        # Convert boolean flag to "BUY" / "SELL"
        if buy_sell_flag_col_name and buy_sell_flag_col_name in df.columns:
            print(f"[{pd.Timestamp.now()}] Converting '{buy_sell_flag_col_name}' column (True->SELL, False->BUY)...")
            df[buy_sell_flag_col_name] = df[buy_sell_flag_col_name].astype(bool)
            df[buy_sell_flag_col_name] = df[buy_sell_flag_col_name].map({True: "SELL", False: "BUY"})
            print(f"[{pd.Timestamp.now()}] Conversion complete.")

        # --- Add Symbol Column ---
        if symbol:
            print(f"[{pd.Timestamp.now()}] Adding 'SYMBOL' column with value '{symbol}'...")
            df['SYMBOL'] = symbol
            # Reorder columns to put SYMBOL at the very beginning
            df_columns = ['SYMBOL'] + [col for col in df.columns if col != 'SYMBOL']
            df = df[df_columns]
            print(f"[{pd.Timestamp.now()}] 'SYMBOL' column added.")

        print(f"[{pd.Timestamp.now()}] Writing processed data to '{output_file_path}'...")
        df.to_csv(
            output_file_path,
            index=include_output_index,
            header=include_output_header
        )
        print(f"[{pd.Timestamp.now()}] Successfully saved to '{output_file_path}'")
        return True

    except FileNotFoundError:
        print(f"[{pd.Timestamp.now()}] Error: The file '{input_file_path}' was not found.")
        return False
    except IndexError:
        print(f"[{pd.Timestamp.now()}] Error: Column index out of range. Check 'column_indices' against your CSV structure and ensure it matches 'new_column_names'.")
        return False
    except ValueError as e:
        print(f"[{pd.Timestamp.now()}] Error processing CSV or arguments: {e}")
        return False
    except Exception as e:
        print(f"[{pd.Timestamp.now()}] An unexpected error occurred: {e}")
        return False

'''
python csv_processor.py BTCUSDT-trades-2025-06-11.csv processed_final.csv "4,0,1,2,5" "TIMESTAMP,ORDER_ID,PRICE,SIZE,BUY_SELL_FLAG" --buy_sell_col BUY_SELL_FLAG --price_to_cents_col PRICE --timestamp_col TIMESTAMP --symbol BTCUSDT'''
# --- Command Line Argument Parsing ---
if __name__ == "__main__":

    data_no_header = """4999132931, 110274.39000000, 0.00290000, 319.79573100, 1749600000041192, True, True
                        4999132932, 110274.40000000, 0.00055000, 60.65092000, 1749600000360424, False, True
                        4999132933, 110274.40000000, 0.00009000, 9.92469600, 1749600000543228, False, True
                        4999132934, 110274.40000000, 0.00181000, 199.59666400, 1749600000622952, False, True
                        4999132935, 110274.39000000, 0.00380000, 419.04268200, 1749600000959833, True, True """

    parser = argparse.ArgumentParser(
        description="Extracts and renames columns from a CSV file, converts timestamps and boolean flags, adds a symbol column, then saves to a new CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path for the output CSV file."
    )
    parser.add_argument(
        "column_indices",
        type=str,
        help="A comma-separated list of 0-based column indices to extract.\n"
             "Example: '4,0,1,2,5' for Trade Time (index 4), Trade ID (index 0), Price (index 1), Quantity (index 2), Is Buyer Maker (index 5)."
    )
    parser.add_argument(
        "new_column_names",
        type=str,
        help="A comma-separated list of new column names for the extracted columns.\n"
             "Must match the order and count of column_indices.\n"
             "Example: 'TIMESTAMP,ORDER_ID,PRICE,SIZE,BUY_SELL_FLAG'."
    )
    parser.add_argument(
        "--no_output_header",
        action="store_true",
        help="Set this flag to prevent writing a header row in the output CSV. (Default: Header is included)"
    )
    parser.add_argument(
        "--include_output_index",
        action="store_true",
        help="Set this flag to include the DataFrame index as the first column in the output CSV. (Default: Index is not included)"
    )
    parser.add_argument(
        "--buy_sell_col",
        type=str,
        help="The new column name that contains the boolean buy/sell flag (e.g., 'BUY_SELL_FLAG').\n"
             "If specified, 'True' (isBuyerMaker) will be converted to 'SELL' and 'False' to 'BUY'."
    )
    parser.add_argument(
        "--price_to_cents_col",
        type=str,
        help="The new column name that holds the price to be multiplied by 100 (e.g., 'PRICE').\n"
             "If specified, this column will be converted to cents."
    )
    parser.add_argument(
        "--timestamp_col",
        type=str,
        help="The new column name that holds the nanosecond Unix timestamp to be converted to datetime (e.g., 'TIMESTAMP')."
    )
    parser.add_argument(
        "--symbol",  # NEW COMMAND-LINE ARGUMENT
        type=str,
        required=True,  # Make it required so user always provides it
        help="The trading symbol (e.g., 'BTCUSDT', 'AAPL') to add as a new 'SYMBOL' column to the output file."
    )

    args = parser.parse_args()

    # Convert comma-separated strings to lists
    try:
        parsed_column_indices = [int(x.strip()) for x in args.column_indices.split(',')]
        parsed_new_column_names = [x.strip() for x in args.new_column_names.split(',')]

        if len(parsed_column_indices) != len(parsed_new_column_names):
            raise ValueError("The number of column indices and new column names must match.")

    except ValueError as e:
        print(f"[{pd.Timestamp.now()}] Error parsing arguments: {e}")
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        print(f"[{pd.Timestamp.now()}] An unexpected error occurred during argument parsing: {e}")
        parser.print_help()
        sys.exit(1)

    # Validate column names passed via arguments
    buy_sell_col_name_in_df = None
    if args.buy_sell_col:
        if args.buy_sell_col in parsed_new_column_names:
            buy_sell_col_name_in_df = args.buy_sell_col
        else:
            print(
                f"[{pd.Timestamp.now()}] Warning: --buy_sell_col '{args.buy_sell_col}' was specified, but it's not in the list of new column names. Buy/Sell conversion will be skipped.")

    price_to_cents_col_name_in_df = None
    if args.price_to_cents_col:
        if args.price_to_cents_col in parsed_new_column_names:
            price_to_cents_col_name_in_df = args.price_to_cents_col
        else:
            print(
                f"[{pd.Timestamp.now()}] Warning: --price_to_cents_col '{args.price_to_cents_col}' was specified, but it's not in the list of new column names. Price conversion will be skipped.")

    timestamp_col_name_in_df = None
    if args.timestamp_col:
        if args.timestamp_col in parsed_new_column_names:
            timestamp_col_name_in_df = args.timestamp_col
        else:
            print(
                f"[{pd.Timestamp.now()}] Warning: --timestamp_col '{args.timestamp_col}' was specified, but it's not in the list of new column names. Timestamp conversion will be skipped.")

    # Call the main processing function
    processing_success = extract_rename_and_save_columns(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        column_indices=parsed_column_indices,
        new_column_names=parsed_new_column_names,
        include_output_header=not args.no_output_header,
        include_output_index=args.include_output_index,
        buy_sell_flag_col_name=buy_sell_col_name_in_df,
        price_to_cents_col_name=price_to_cents_col_name_in_df,
        timestamp_col_name=timestamp_col_name_in_df,
        symbol=args.symbol  # Pass the new symbol argument
    )

    if processing_success:
        print(f"[{pd.Timestamp.now()}] Script finished successfully.")
        sys.exit(0)
    else:
        print(f"[{pd.Timestamp.now()}] Script failed.")
        sys.exit(1)