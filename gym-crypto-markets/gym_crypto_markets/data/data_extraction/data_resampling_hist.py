import pandas as pd
import numpy as np
import argparse
import sys
from typing import List


def process_trade_data_for_env(
        input_file: str,
        output_file: str,
        freq: str = '1min',
        scale_price: bool = False
):
    """
    Processes a raw trade data CSV into a format suitable for the HistoricalTradingEnv.

    This function reads a tick-by-tick trade log, aggregates it into fixed
    time intervals (e.g., 1 minute), and calculates the close price, VWAP,
    total volume, and buy volume for each interval.

    Args:
        input_file: Path to the raw input CSV file.
        output_file: Path to save the processed output file.
        freq: The frequency to resample the data (default: '1min').
        scale_price: If True, scales the price by 100 to represent cents.

    Returns:
        True if processing was successful, False otherwise.
    """
    try:
        # --- 1. Load Raw Trade Data ---
        print(f"Loading raw trade data from {input_file}...")
        # These are the expected columns from a typical crypto exchange data feed.
        # 'is_buyer_maker': False means the taker was a buyer.
        col_names: List[str] = ['price', 'quantity', 'timestamp', 'is_buyer_maker']
        col_indices = [1, 2, 4, 5]
        df = pd.read_csv(
            input_file,
            header=0,
            usecols=col_indices,
            names=col_names
        )

        # --- 2. Pre-processing ---
        print("Converting timestamp from unit 'ms' to datetime objects...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', errors='raise')

        # Ensure numeric types for calculation
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(inplace=True)

        if scale_price:
            print("Scaling price to notional cents for ABIDES...")
            df['price'] = (df['price'] * 100).round(4)

        # Create helper columns for aggregation
        # 'price_x_quantity' is used to calculate VWAP efficiently
        df['price_x_quantity'] = df['price'] * df['quantity']

        # 'buy_volume' is the quantity of trades where the taker was a buyer.
        # In many data sources, is_buyer_maker=False indicates a market buy.
        df['buy_volume'] = df.apply(
            lambda row: row['quantity'] if not row['is_buyer_maker'] else 0, axis=1
        )

        df.set_index('timestamp', inplace=True)

        # --- 3. Resample and Aggregate Data ---
        print(f"Resampling data to '{freq}' frequency...")

        # Define the aggregation logic for each column
        aggregation_rules = {
            'price': 'last',  # Closing price for the interval
            'quantity': 'sum',  # Total volume for the interval
            'buy_volume': 'sum',  # Total buy volume for the interval
            'price_x_quantity': 'sum'  # Sum of (price*qty) for VWAP calculation
        }

        resampled_df = df.resample(freq).agg(aggregation_rules)

        # --- 4. Post-processing and Final Calculations ---
        resampled_df['vwap'] = resampled_df['price_x_quantity'] / resampled_df['quantity']

        # Rename 'quantity' to 'volume' to match the required format
        resampled_df.rename(columns={'quantity': 'volume'}, inplace=True)

        # Select and reorder columns, creating an explicit copy to avoid warnings
        final_df = resampled_df[['price', 'vwap', 'volume', 'buy_volume']].copy()

        # Now, operations on final_df are safe
        final_df['vwap'].fillna(method='ffill', inplace=True)
        final_df['vwap'].fillna(method='bfill', inplace=True)

        # Forward-fill any empty intervals (minutes with no trades)
        print("Forward-filling any missing intervals...")
        final_df.ffill(inplace=True)
        final_df.dropna(inplace=True)  # Drop any remaining NaNs (e.g., at the start)

        if final_df.empty:
            print("Error: The final data series is empty after processing.", file=sys.stderr)
            return False

        # --- 5. Save the Processed Data ---
        print(f"Saving processed data to {output_file}...")
        final_df.to_csv(output_file, header=True, float_format='%.4f')
        print("Processing complete.")
        return True

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        return False
    except KeyError as e:
        print(f"Error: Missing required column in input file: {e}. Expected {col_indices}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process raw trade data into 1-minute bars for the historical trading environment."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the raw input CSV. Must contain 'timestamp', 'price', 'quantity', 'is_buyer_maker' columns."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the processed 1-minute data file."
    )
    parser.add_argument(
        "--freq",
        type=str,
        default='1min',
        help="Frequency to resample data (default: '1min')."
    )
    parser.add_argument(
        "--scale_price",
        action="store_true",
        help="Include this flag to scale the price by 100 (e.g., to cents)."
    )
    args = parser.parse_args()

    if process_trade_data_for_env(
            input_file=args.input_file,
            output_file=args.output_file,
            freq=args.freq,
            scale_price=args.scale_price
    ):
        sys.exit(0)
    else:
        sys.exit(1)
