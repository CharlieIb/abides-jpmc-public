import pandas as pd
import argparse # Import the argparse library

def process_trade_data(input_file: str, output_file: str, freq: str):
    """
    Processes raw trade data into a fixed-interval price series.
    """
    print(f"Loading raw trade data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['TIMESTAMP'])
    df.set_index('TIMESTAMP', inplace=True)

    print(f"Resampling data to '{freq}' frequency...")
    # Resample the data and get the last price in each interval
    price_series = df['PRICE'].resample(freq).last()

    print("Forward-filling missing values...")
    # Forward-fill any empty intervals
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