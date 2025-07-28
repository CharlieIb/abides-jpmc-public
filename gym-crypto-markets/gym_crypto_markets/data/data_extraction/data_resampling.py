import pandas as pd
import argparse
import sys

def resample_price_data(
    input_file: str,
    output_file: str,
    freq: str,
    scale_price: bool
):

    try:


        print(f"Loading simplified data from {input_file}...")
        df = pd.read_csv(
            input_file,
            usecols=[1, 4],
            header=None,
            names=['PRICE', 'TIMESTAMP']
        )

        print("Converting timestamp from unit 'us'...")
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='us', errors='raise')

        df.set_index('TIMESTAMP', inplace=True)

        print(df.head())

        if scale_price:
            print("Scaling price to notional cents for ABIDES...")
            df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
            df['PRICE'] = (df['PRICE'] * 100).round(4)

        print(f"Resampling data to '{freq}' frequency...")
        price_series = df['PRICE'].resample(freq).last()

        print("Forward-filling any missing values...")
        price_series.ffill(inplace=True)
        price_series.dropna(inplace=True)

        if price_series.empty:
            print("Error: The final data series is empty after processing.", file=sys.stderr)
            return False

        print(f"Saving processed data to {output_file}...")
        price_series.to_csv(output_file, header=True, float_format='%.4f')
        print("Processing complete.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Resample a simplified two-column CSV file."
    )
    parser.add_argument("input_file", type=str, help="Path to the simplified input CSV (timestamp, price).")
    parser.add_argument("output_file", type=str, help="Path to save the processed output file.")
    parser.add_argument("freq", type=str, help="Frequency to resample data (e.g., '1S').")
    parser.add_argument("--scale_price", action="store_true", help="Include flag to scale the price.")
    args = parser.parse_args()

    if resample_price_data(
        input_file=args.input_file,
        output_file=args.output_file,
        freq=args.freq,
        scale_price=args.scale_price
    ):
        sys.exit(0)
    else:
        sys.exit(1)