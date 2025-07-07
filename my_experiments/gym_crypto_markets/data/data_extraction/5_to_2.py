import pandas as pd
import argparse # Import the argparse library

def process_trade_data(input_file: str, output_file: str, freq: str):
    """
    Processes raw trade data into a fixed-interval price series.
    """
    print(f"Loading raw trade data from {input_file}...")
    df = pd.read_csv(input_file)
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='ns')
    else:
        raise ValueError("TIMESTAMP column not found in input CSV.")

    df.set_index('TIMESTAMP', inplace=True)

    print(f"Resampling data to '{freq}' frequency...")
    price_series = df['PRICE'].resample(freq).last()

    # Check for gaps after resampling but before forward-fill
    full_index = pd.date_range(start=price_series.index.min(), end=price_series.index.max(), freq=freq)
    missing_timestamps = full_index.difference(price_series.index)

    if missing_timestamps.empty:
        print("No missing timestamps after resampling.")
    else:
        print(f"Warning: Missing timestamps detected: {len(missing_timestamps)}")
        print(missing_timestamps)

    print("Forward-filling missing values...")
    price_series = price_series.ffill()
    price_series.dropna(inplace=True)

    print(f"Saving processed data to {output_file}...")
    price_series.to_csv(output_file, header=True)
    print("Processing complete.")
# --- Main execution block ---
if __name__ == '__main__':
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Resample trade data to a specified frequency.")

    # 2. Define the command-line arguments
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file with raw trades.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the processed output CSV file.')
    parser.add_argument('--freq', type=str, default='1Min', help="Frequency to resample the data (e.g., '1S', '5Min', '100L' for milliseconds). Default is '1Min'.")

    # 3. Parse the arguments provided by the user
    args = parser.parse_args()

    # 4. Call the function with the user-provided arguments
    process_trade_data(input_file=args.input, output_file=args.output, freq=args.freq)