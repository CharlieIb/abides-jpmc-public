import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile


#  Configuration
# --- MODIFICATION: This is now a list to hold multiple file paths ---
HISTORICAL_DATA_PATHS = [
    '/home/charlie/PycharmProjects/ABIDES_GYM_EXT/abides-jpmc-public/gym-crypto-markets/gym_crypto_markets/data/data_analysis/BTCUSDT-trades-2025-07-17.csv'
    ]

# HISTORICAL_DATA_PATHS = [
#     'BTCUSDT-trades-2020-01-20.csv',
#     'BTCUSDT-trades-2020-02-03.csv',
#     'BTCUSDT-trades-2020-03-01.csv',
#     'BTCUSDT-trades-2020-03-30.csv',
#     'BTCUSDT-trades-2020-04-17.csv',
#     'BTCUSDT-trades-2020-05-11.csv',
#     'BTCUSDT-trades-2020-05-31.csv',
#     'BTCUSDT-trades-2020-06-20.csv',
#     'BTCUSDT-trades-2020-07-10.csv',
#     'BTCUSDT-trades-2020-07-31.csv',
#     'BTCUSDT-trades-2020-08-14.csv',
# ]

# Note: This assumes the CSV file inside any ZIP archive has this name.
CSV_FILE_IN_ZIP = 'BTCUSDT-trades-2025-05.csv'
QUANTITY_COLUMN_INDEX = 2
QUANTITY_COLUMN_NAME = 'quantity'

NOTIONAL_SCALING_FACTOR = 100000

NOISE_MAX_QTY = 25
VALUE_MIN_QTY = 100
MOMENTUM_MIN_QTY = 0

# Define categories/bins for trade quantities
QUANTITY_BINS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, np.inf]
QUANTITY_LABELS = ['<0.00001(<1)', '0.00001-0.0001(1-10)', '0.0001-0.001(10-100)', '0.001-0.01(100-1000',
                   '0.01-0.1(1000-10000)', '0.1-1(10000-100000)', '1-5', '5-10', '10-50', '50+']

QUANTITY_BINS_SCALED = [q * NOTIONAL_SCALING_FACTOR for q in QUANTITY_BINS[:-1]] + [np.inf]


# --- MODIFIED: Data Loading Function ---
def load_and_combine_historical_data(file_paths: list, csv_in_zip_name: str, column_index: int,
                                     column_name: str) -> pd.DataFrame:
    """
    Loads and combines historical trade data from multiple CSV or ZIP files.
    """
    all_dataframes = []

    print(f"--- Starting to load {len(file_paths)} data file(s) ---")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found at {file_path}. Skipping.")
            continue

        try:
            temp_df = None
            # Determine if the file is a ZIP archive
            if file_path.lower().endswith('.zip'):
                print(f"Detected ZIP file: {file_path}. Reading '{csv_in_zip_name}'...")
                with zipfile.ZipFile(file_path, 'r') as zf:
                    if csv_in_zip_name not in zf.namelist():
                        print(f"Error: '{csv_in_zip_name}' not found inside '{file_path}'. Skipping.")
                        continue
                    with zf.open(csv_in_zip_name) as f:
                        temp_df = pd.read_csv(f, header=None)
            else:
                print(f"Detected CSV file: {file_path}. Reading directly...")
                temp_df = pd.read_csv(file_path, header=None)

            print(f"  > Successfully loaded data from {os.path.basename(file_path)}. Shape: {temp_df.shape}")

            # Check if the specified column index exists
            if column_index >= temp_df.shape[1]:
                print(
                    f"Error: Column index {column_index} is out of bounds for {os.path.basename(file_path)}. Skipping.")
                continue

            # Rename the specified column
            temp_df.rename(columns={column_index: column_name}, inplace=True)
            all_dataframes.append(temp_df)

        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping.")
            continue

    if not all_dataframes:
        print("\nError: No data could be loaded from any of the provided file paths.")
        return pd.DataFrame()

    # Combine all loaded dataframes into one
    print(f"\n--- Combining {len(all_dataframes)} loaded data file(s) ---")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Successfully combined data. Total shape: {combined_df.shape}")

    return combined_df


# Data Analysis (No changes needed in this function)
def analyze_quantities(df: pd.DataFrame, quantity_col: str):
    """
    Performs descriptive statistics and categorization on trade quantities.
    """
    if df.empty or quantity_col not in df.columns:
        print(f"Error: '{quantity_col}' column not found or DataFrame is empty. Cannot analyze quantities.")
        return

    print(f"\n--- Analyzing Trade Quantities in '{quantity_col}' Column ---")

    # Convert to numeric, handling potential errors
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')
    df.dropna(subset=[quantity_col], inplace=True)  # Remove rows where quantity couldn't be converted
    NOTIONAL_SCALING_FACTOR = 100000

    print(f"\nApplying notional scaling factor of {NOTIONAL_SCALING_FACTOR} to quantities.")
    df['notional_quantity'] = df[quantity_col] * NOTIONAL_SCALING_FACTOR  # Create new column for clarity

    # Calculate overall Log-Normal parameters for the scaled data
    scaled_quantities_for_log_overall = df[df['notional_quantity'] > 0]['notional_quantity']
    log_scaled_quantities_overall = np.log(scaled_quantities_for_log_overall)
    log_mean_scaled_overall = log_scaled_quantities_overall.mean()
    log_std_scaled_overall = log_scaled_quantities_overall.std()
    print(
        f"\nCalculated Overall Log-Normal parameters for scaled data: Mean={log_mean_scaled_overall:.4f}, Std={log_std_scaled_overall:.4f}")

    # Basic statistics for Notional Quantities
    print("\nDescriptive Statistics for Notional Quantities (Overall):")
    print(df['notional_quantity'].describe())

    max_qty_notional = df['notional_quantity'].max()
    min_qty_notional = df['notional_quantity'].min()
    print(f"\nMaximum Notional Quantity Traded (Overall): {max_qty_notional}")
    print(f"Minimum Notional Quantity Traded (Overall): {min_qty_notional}")

    # Categorize quantities into bins
    df['quantity_category'] = pd.cut(df['notional_quantity'], bins=QUANTITY_BINS_SCALED, labels=QUANTITY_LABELS,
                                     right=True, include_lowest=True)

    print("\nNotional Quantity Distribution by Category (Overall):")
    category_counts = df['quantity_category'].value_counts(sort=False)
    category_percentages = df['quantity_category'].value_counts(normalize=True, sort=False) * 100

    category_df = pd.DataFrame({
        'Count': category_counts,
        'Percentage': category_percentages
    })
    print(category_df)

    # --- NEW ANALYSIS SECTION ---
    print("\n--- Analysis of Predefined Trade Amount Frequencies ---")

    # Define the specific notional amounts you are interested in
    target_amounts = [50, 100, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]

    # Define the percentage range to look around each target amount (e.g., +/- 5%)
    percentage_range = 0.025

    amount_frequencies = []

    for amount in target_amounts:
        lower_bound = amount * (1 - percentage_range)
        upper_bound = amount * (1 + percentage_range)

        # Filter the DataFrame to find trades within this range
        trades_in_range = df[
            (df['notional_quantity'] >= lower_bound) &
            (df['notional_quantity'] <= upper_bound)
            ]

        count = len(trades_in_range)

        if count > 0:
            percentage_of_total = (count / len(df)) * 100
            amount_frequencies.append({
                'Target Amount': amount,
                'Range (Notional)': f"{lower_bound:.2f} - {upper_bound:.2f}",
                'Frequency (Count)': count,
                'Percentage of Total Trades': f"{percentage_of_total:.4f}%"
            })

    if amount_frequencies:
        # Create a DataFrame from the results and sort by frequency
        freq_df = pd.DataFrame(amount_frequencies)
        freq_df.sort_values(by='Frequency (Count)', ascending=False, inplace=True)

        print(f"Frequencies of trades around target amounts (within a +/- {percentage_range * 100}% range):")
        print(freq_df.to_string(index=False))
    else:
        print("No trades found around the specified target amounts.")

    print("\n--- Inferring Agent Trade Types and Analyzing Filtered Quantities ---")

    # Noise Agent Quantities
    noise_quantities = df[df['notional_quantity'] <= NOISE_MAX_QTY]['notional_quantity']
    print(f"\n--- Analysis for Inferred NOISE Agent Quantities (<= {NOISE_MAX_QTY} Notional) ---")
    if not noise_quantities.empty:
        print("Descriptive Statistics:")
        print(noise_quantities.describe())
        log_noise_quantities = np.log(noise_quantities[noise_quantities > 0])
        if not log_noise_quantities.empty:
            print(
                f"  Log-Normal Parameters (Mean of Log, Std of Log): Mean={log_noise_quantities.mean():.4f}, Std={log_noise_quantities.std():.4f}")
        else:
            print("  No positive quantities found for Log-Normal parameter calculation for Noise agents.")
    else:
        print("  No trades found for Noise agent analysis in this range.")

    # Momentum Agent Quantities
    momentum_quantities = df[df['notional_quantity'] >= MOMENTUM_MIN_QTY]['notional_quantity']
    print(f"\n--- Analysis for Inferred MOMENTUM Agent Quantities (>= {MOMENTUM_MIN_QTY} Notional) ---")
    if not momentum_quantities.empty:
        print("Descriptive Statistics:")
        print(momentum_quantities.describe())
        log_momentum_quantities = np.log(momentum_quantities[momentum_quantities > 0])
        if not log_momentum_quantities.empty:
            print(
                f"  Log-Normal Parameters (Mean of Log, Std of Log): Mean={log_momentum_quantities.mean():.4f}, Std={log_momentum_quantities.std():.4f}")
        else:
            print("  No positive quantities found for Log-Normal parameter calculation for Momentum agents.")
    else:
        print("  No trades found for Momentum agent analysis in this range.")

    # Value Agent Quantities
    value_quantities = df[df['notional_quantity'] >= VALUE_MIN_QTY]['notional_quantity']
    print(f"\n--- Analysis for Inferred VALUE Agent Quantities (>= {VALUE_MIN_QTY} Notional) ---")
    if not value_quantities.empty:
        print("Descriptive Statistics:")
        print(value_quantities.describe())
        log_value_quantities = np.log(value_quantities[value_quantities > 0])
        if not log_value_quantities.empty:
            print(
                f"  Log-Normal Parameters (Mean of Log, Std of Log): Mean={log_value_quantities.mean():.4f}, Std={log_value_quantities.std():.4f}")
        else:
            print("  No positive quantities found for Log-Normal parameter calculation for Value agents.")
    else:
        print("  No trades found for Value agent analysis in this range.")

    # Analysis of Specific Notional Quantity Clusters
    print("\n--- Analysis of Specific Notional Quantity Clusters (for Normal Distributions) ---")

    # Define the clusters you suspect. Adjust these ranges based on your histogram observations.
    # These are notional quantities.
    clusters_to_analyze = {
        "Cluster 80-120": (80, 120),
        "Cluster ~500": (450, 550),
        "Cluster ~900-1050": (900, 1050),
        "Cluster ~1800-2200": (1800, 2200),
        "Cluster 2500": (2400, 2600),
        "Cluster 5000": (4500, 5500),
        "Cluster 10K (0.1 BTC)": (9750, 10250),
        "Cluster 100K (1 BTC)": (99000, 110000),
        "Cluster 1M (10 BTC)": (900000, 1100000),
    }

    total_trades_in_clusters = 0
    for cluster_name, (lower_bound, upper_bound) in clusters_to_analyze.items():
        cluster_data = df[(df['notional_quantity'] >= lower_bound) & (df['notional_quantity'] <= upper_bound)][
            'notional_quantity']

        if not cluster_data.empty:
            count = len(cluster_data)
            percentage = (count / len(df)) * 100
            total_trades_in_clusters += count

            print(f"\n--- {cluster_name} (Range: {lower_bound}-{upper_bound} Notional) ---")
            print(f"  Count: {count}, Percentage of Total Trades: {percentage:.4f}%")
            print("  Descriptive Statistics:")
            print(cluster_data.describe())

            # Calculate Log-Normal parameters for this cluster
            log_cluster_data = np.log(cluster_data[cluster_data > 0])
            if not log_cluster_data.empty:
                print(
                    f"  Log-Normal Parameters (Mean of Log, Std of Log): Mean={log_cluster_data.mean():.4f}, Std={log_cluster_data.std():.4f}")
            else:
                print("  No positive quantities found for Log-Normal parameter calculation in this cluster.")
        else:
            print(f"\n--- {cluster_name} (Range: {lower_bound}-{upper_bound} Notional) ---")
            print("  No trades found in this cluster range.")

    print(
        f"\nTotal trades found in all specified clusters: {total_trades_in_clusters} ({total_trades_in_clusters / len(df) * 100:.4f}% of total).")

    print("\nAnalysis complete. Use these insights to refine your OrderSizeModel.")


# --- MODIFIED: Main Execution ---
if __name__ == "__main__":
    # Call the new function with the list of paths
    combined_trades_df = load_and_combine_historical_data(
        HISTORICAL_DATA_PATHS,
        CSV_FILE_IN_ZIP,
        QUANTITY_COLUMN_INDEX,
        QUANTITY_COLUMN_NAME
    )

    # The rest of the script runs on the combined DataFrame
    if not combined_trades_df.empty:
        analyze_quantities(combined_trades_df, QUANTITY_COLUMN_NAME)