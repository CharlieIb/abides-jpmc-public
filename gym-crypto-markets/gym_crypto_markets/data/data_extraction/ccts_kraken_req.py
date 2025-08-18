import ccxt
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime, timedelta, timezone
import time
from typing import List, Optional


#  Function to fetch raw Kraken trade data using CCXT
def get_kraken_trades_for_day_ccxt(symbol: str, date_str: str) -> pd.DataFrame:
    """
    Fetches all trade data for a specific day from Kraken using CCXT.

    Args:
        symbol (str): The trading pair (e.g., 'BTC/USDT'). CCXT uses unified symbols.
        date_str (str): The date in 'YYYY-MM-DD' format.
                        Data will be fetched for this date in UTC.

    Returns:
        pd.DataFrame: DataFrame containing raw trade data:
                      'timestamp' (datetime, UTC), 'price' (float),
                      'quantity' (float), 'is_buy' (bool).
                      Returns an empty DataFrame if an error occurs or no data is found.
    """
    kraken = ccxt.kraken({
        'enableRateLimit': True,  # Enable CCXT's built-in rate limiter
        'timeout': 15000,  # Set a timeout for API calls (15 seconds)
    })

    all_trades = []
    MAX_RETRIES = 5
    RETRY_DELAY_SECONDS = 5

    try:
        start_of_day_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_of_day_dt = start_of_day_dt + timedelta(days=1)
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD.")
        return pd.DataFrame()

    start_of_day_ms = int(start_of_day_dt.timestamp() * 1000)
    end_of_day_ms = int(end_of_day_dt.timestamp() * 1000)

    since = start_of_day_ms
    limit = 1000  # Max trades per request for Kraken's fetch_trades

    print(f"Attempting to fetch raw trade data for {symbol} on {date_str} (UTC) using CCXT...")

    try:
        kraken.load_markets()
        if symbol not in kraken.symbols:
            print(f"Error: Symbol '{symbol}' not recognized by Kraken CCXT adapter.")
            # Try common alternatives for BTC/USDT or BTC/USD on Kraken
            if 'XBT/USDT' in kraken.symbols and symbol == 'BTC/USDT':
                print(f"Falling back to 'XBT/USDT' for Kraken.")
                symbol = 'XBT/USDT'
            elif 'XBT/USD' in kraken.symbols and symbol == 'BTC/USD':
                print(f"Falling back to 'XBT/USD' for Kraken.")
                symbol = 'XBT/USD'
            else:
                print(f"Available symbols for BTC/XBT: {[s for s in kraken.symbols if 'BTC' in s or 'XBT' in s]}")
                return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Kraken markets: {e}", file=sys.stderr)
        return pd.DataFrame()

    while True:
        retries = 0
        trades_batch = []  # Renamed to avoid conflict with outer 'trades' list if any
        while retries < MAX_RETRIES:
            try:
                # fetch_trades returns data sorted by timestamp ascending
                trades_batch = kraken.fetch_trades(symbol, since=since, limit=limit)
                break  # Break from retry loop if successful
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"CCXT Error fetching trades: {e}. Retrying ({retries + 1}/{MAX_RETRIES})...", file=sys.stderr)
                retries += 1
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                print(f"An unexpected error during CCXT fetch: {e}. Aborting.", file=sys.stderr)
                return pd.DataFrame()

        if retries == MAX_RETRIES:
            print(f"Failed to fetch data after {MAX_RETRIES} retries. Stopping.", file=sys.stderr)
            break  # Break from main fetching loop

        if not trades_batch:
            print("No more trades found or reached end of available data from API.")
            break  # No more trades to fetch

        trades_added_in_batch = 0
        for trade in trades_batch:
            trade_timestamp_ms = trade['timestamp']  # CCXT normalizes timestamp to milliseconds

            # Only add trades strictly within the target day
            if start_of_day_ms <= trade_timestamp_ms < end_of_day_ms:
                all_trades.append({
                    'timestamp': trade_timestamp_ms,  # Storing in milliseconds
                    'price': float(trade['price']),
                    'quantity': float(trade['amount']),  # 'amount' is the quantity
                    'is_buy': (trade['side'] == 'buy')  # 'buy' or 'sell'
                })
                trades_added_in_batch += 1
            elif trade_timestamp_ms >= end_of_day_ms:
                # We've fetched trades beyond our target day, stop
                print(f"Fetched trades up to or past end of {date_str}.")
                break  # Break from inner for-loop (trades_batch loop)

        # If the last trade in the *current batch* (regardless of whether it was added)
        # is already at or past the end of our target day, then we're done.
        # This check is crucial for exiting the outer while loop.
        if trades_batch and trades_batch[-1]['timestamp'] >= end_of_day_ms:
            print(f"Reached data beyond {date_str}. Stopping.")
            break

        # If no trades were added in the batch, but trades_batch was not empty (meaning they were all past our date),
        # then the outer loop should have broken. If it didn't, and no new trades were added
        # (meaning the trades_batch were all *before* our start_of_day or there was a gap),
        # but trades_batch itself was not empty, then we still need to advance 'since'.
        if trades_batch:
            since = trades_batch[-1]['timestamp'] + 1  # Advance 'since' to the next millisecond after the last trade
        else:
            # If trades_batch list is empty after a fetch, and we didn't break earlier, means no more data
            break

        # Print progress
        if len(all_trades) % 5000 == 0:
            print(f"Fetched {len(all_trades)} trades so far for {symbol}...")

    df = pd.DataFrame(all_trades)
    if not df.empty:
        # Convert timestamp to datetime and ensure UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        # Final filter to ensure data is strictly within the requested day in UTC
        df = df[(df['timestamp'] >= start_of_day_dt) & (df['timestamp'] < end_of_day_dt)]

    print(f"Finished fetching raw data. Total trades retrieved for {symbol} on {date_str}: {len(df)}")
    return df


# --- Modified function to process the DataFrame from CCXT ---
def process_dataframe_for_env(
        df_raw_trades: pd.DataFrame,
        output_file: str,
        freq: str = '1min',
        scale_price: bool = False,
        normalize_quantity_factor: float = 1.0  # New parameter for quantity normalization
) -> bool:
    """
    Processes a DataFrame of raw trade data into a format suitable for the HistoricalTradingEnv.

    This function expects a DataFrame with columns 'timestamp' (datetime), 'price',
    'quantity', and 'is_buy' (boolean). It aggregates data into fixed
    time intervals (e.g., 1 minute), and calculates the close price, VWAP,
    total volume, and buy volume for each interval.

    Args:
        df_raw_trades: DataFrame containing raw trade data. Expected columns:
                       'timestamp' (already a datetime object), 'price',
                       'quantity', 'is_buy'.
        output_file: Path to save the processed output file.
        freq: The frequency to resample the data (default: '1min').
        scale_price: If True, scales the price by 100 to represent cents.
        normalize_quantity_factor: Factor to multiply the quantity by for normalization.

    Returns:
        True if processing was successful, False otherwise.
    """
    try:
        if df_raw_trades.empty:
            print("Input DataFrame is empty. Nothing to process.", file=sys.stderr)
            return False

        print("Starting processing of raw trade data into time bars...")

        # --- 1. Initial Data Preparation ---
        df = df_raw_trades.copy()  # Work on a copy to avoid modifying original

        # Ensure 'timestamp' is already a datetime and has timezone info
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')  # Coerce errors just in case
        elif df['timestamp'].dt.tz is None:  # If datetime but naive, localize to UTC
            df['timestamp'] = df['timestamp'].dt.tz_localize(timezone.utc)

        df.set_index('timestamp', inplace=True)

        # Ensure numeric types
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['price', 'quantity'], inplace=True)  # Drop rows where price or quantity became NaN

        # --- Apply Quantity Normalization ---
        if normalize_quantity_factor != 1.0:
            print(f"Normalizing quantity by factor of {normalize_quantity_factor}...")
            df['quantity'] = df['quantity'] * normalize_quantity_factor

        if scale_price:
            print("Scaling price to notional cents for ABIDES...")
            df['price'] = (df['price'] * 100).round(0)

        # Create helper columns for aggregation
        df['price_x_quantity'] = df['price'] * df['quantity']

        # 'buy_volume': quantity where 'is_buy' is True (taker was a buyer)
        df['buy_volume'] = df.apply(
            lambda row: row['quantity'] if row['is_buy'] else 0, axis=1
        )

        # --- 2. Resample and Aggregate Data ---
        print(f"Resampling data to '{freq}' frequency...")

        aggregation_rules = {
            'price': 'last',  # Closing price for the interval
            'quantity': 'sum',  # Total volume for the interval
            'buy_volume': 'sum',  # Total buy volume for the interval
            'price_x_quantity': 'sum'  # Sum of (price*qty) for VWAP calculation
        }

        # Apply resampling. Use origin='start' to ensure intervals align consistently.
        resampled_df = df.resample(freq, origin='start').agg(aggregation_rules)

        # --- 3. Post-processing and Final Calculations ---
        # Handle cases where quantity is zero (no trades in an interval) to avoid division by zero
        resampled_df['vwap'] = np.where(
            resampled_df['quantity'] > 0,
            resampled_df['price_x_quantity'] / resampled_df['quantity'],
            np.nan  # Set VWAP to NaN if no trades
        )

        resampled_df.rename(columns={'quantity': 'volume'}, inplace=True)

        final_df = resampled_df[['price', 'vwap', 'volume', 'buy_volume']].copy()

        # Fill missing values
        print("Filling missing values for price, VWAP, and setting zero for volume/buy_volume...")
        # Fill price first, as VWAP depends on it being available in ffill scenario
        final_df['price'].ffill(inplace=True)
        final_df['price'].bfill(inplace=True)  # For any NaNs at the very start

        final_df['vwap'].fillna(method='ffill', inplace=True)
        final_df['vwap'].fillna(method='bfill', inplace=True)  # To handle NaNs at the very start

        final_df['volume'].fillna(0, inplace=True)  # No trades means 0 volume
        final_df['buy_volume'].fillna(0, inplace=True)  # No trades means 0 buy volume

        # Drop any remaining NaNs that couldn't be filled (e.g., if entire series is NaN)
        final_df.dropna(inplace=True)

        if final_df.empty:
            print("Error: The final data series is empty after processing. "
                  "This might happen if no valid trades were found or after filling.", file=sys.stderr)
            return False

        # --- 4. Save the Processed Data ---
        print(f"Saving processed data to {output_file}...")
        final_df.index.name = 'timestamp'  # Ensure the index (timestamp) column is named
        final_df.to_csv(output_file, header=True, float_format='%.4f')
        print("Processing complete.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        return False


# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fetch Kraken trade data via CCXT and process into 1-minute bars for the trading environment."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default='BTC/USDT',
        help="Trading pair symbol (e.g., 'BTC/USDT')."
    )
    parser.add_argument(
        "--date",
        type=str,
        # Default to yesterday's date in UTC. Current time in London is July 31, 2025
        default=(datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d'),
        help="Date to fetch data for in YYYY-MM-DD format (UTC). Default is yesterday."
    )
    parser.add_argument(
        "output_file",  # Positional argument
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
    parser.add_argument(
        "--normalize_quantity",
        type=float,
        default=1.0,  # Default is no normalization (factor of 1.0)
        help="Factor to normalize the quantity by (e.g., 100000 for 1/0.00001)."
    )
    args = parser.parse_args()

    # Step 1: Fetch raw trade data using CCXT
    raw_trades_df = get_kraken_trades_for_day_ccxt(args.symbol, args.date)

    if raw_trades_df.empty:
        print("Failed to retrieve raw trade data or no trades found for the specified day. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Process the retrieved DataFrame
    if process_dataframe_for_env(
            df_raw_trades=raw_trades_df,
            output_file=args.output_file,
            freq=args.freq,
            scale_price=args.scale_price,
            normalize_quantity_factor=args.normalize_quantity  # Pass the new argument
    ):
        sys.exit(0)
    else:
        sys.exit(1)