import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile


# --- Configuration ---
# IMPORTANT: Replace 'path/to/your/historical_trades.csv' with the actual path to your data file.
# This file should ideally contain a column with trade quantities.
HISTORICAL_DATA_PATH = '/gym_crypto_markets/data/data_extraction/archive/BTCUSDT-trades-2025-06-11.csv'
#HISTORICAL_DATA_PATH = '/rds/projects/a/aranboll-ai-research/abides_gym_crypto_sim/abides-jpmc-public/gym-crypto-markets/gym_crypto_markets/data/data_extraction/BTCUSDT-trades-2025-05.zip'
CSV_FILE_IN_ZIP = 'BTCUSDT-trades-2025-05.csv'
QUANTITY_COLUMN_INDEX = 2
QUANTITY_COLUMN_NAME = 'quantity' # Replace with the actual column name for trade quantities in your CSV

NOTIONAL_SCALING_FACTOR = 100000

NOISE_MAX_QTY = 25
VALUE_MIN_QTY = 100
MOMENTUM_MIN_QTY = 10

# Define categories/bins for trade quantities, adjusted for Bitcoin (fractional quantities).
# These bins are designed to capture the typical range from very small to larger quantities.
# You might need to fine-tune these further after an initial run with your actual data.
QUANTITY_BINS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, np.inf]
QUANTITY_LABELS = ['<0.00001(<1)', '0.00001-0.0001(1-10)', '0.0001-0.001(10-100)', '0.001-0.01(100-1000', '0.01-0.1(1000-10000)', '0.1-1(10000-100000)', '1-5', '5-10', '10-50', '50+']

QUANTITY_BINS_SCALED = [q * NOTIONAL_SCALING_FACTOR for q in QUANTITY_BINS[:-1]] + [np.inf]


# --- Data Loading ---
def load_historical_data(file_path: str, csv_in_zip_name: str, column_index: int, column_name: str) -> pd.DataFrame:
    """
    Loads historical trade data from a CSV file.
    Assumes the file is a CSV. Adjust if your data is in a different format (e.g., Excel, Parquet).
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("Please update HISTORICAL_DATA_PATH to the correct location.")
        return pd.DataFrame()

    try:
        # Determine if the file is a ZIP archive
        if file_path.lower().endswith('.zip'):
            print(f"Detected ZIP file: {file_path}. Attempting to read '{csv_in_zip_name}' from it.")
            with zipfile.ZipFile(file_path, 'r') as zf:
                if csv_in_zip_name not in zf.namelist():
                    print(f"Error: '{csv_in_zip_name}' not found inside '{file_path}'.")
                    print("Please update CSV_FILE_IN_ZIP to the correct file name within the archive.")
                    return pd.DataFrame()
                with zf.open(csv_in_zip_name) as f:
                    df = pd.read_csv(f, header=None)
        else:
            print(f"Detected CSV file: {file_path}. Reading directly.")
            df = pd.read_csv(file_path, header=None)

        print(f"Successfully loaded data. Shape: {df.shape}")

        # Check if the specified column index exists
        if column_index >= df.shape[1]:
            print(
                f"Error: Column index {column_index} is out of bounds for the loaded data (only {df.shape[1]} columns).")
            return pd.DataFrame()

        # Rename the specified column to the internal QUANTITY_COLUMN_NAME
        df.rename(columns={column_index: column_name}, inplace=True)

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Data Analysis ---
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
    df.dropna(subset=[quantity_col], inplace=True) # Remove rows where quantity couldn't be converted
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

    print("\n--- Inferring Agent Trade Types and Analyzing Filtered Quantities ---")

    # 1. Noise Agent Quantities: All trades up to NOISE_MAX_QTY
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

    # 2. Momentum Agent Quantities: Trades >= MOMENTUM_MIN_QTY
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

    # 3. Value Agent Quantities: Trades >= VALUE_MIN_QTY
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

    # --- Analysis of Specific Notional Quantity Clusters ---
    print("\n--- Analysis of Specific Notional Quantity Clusters (for Normal Distributions) ---")

    # Define the clusters you suspect. Adjust these ranges based on your histogram observations.
    # These are notional quantities.
    clusters_to_analyze = {
        "Cluster 50-70": (50, 70),
        "Cluster ~100": (90, 110),
        "Cluster ~200-250": (190, 210),
        "Cluster ~240-260": (240, 260),
        "Cluster 500": (450, 550),
        "Cluster 1000": (950, 1050),
        "Cluster 100k (1 BTC)": (90000, 110000),
        "Cluster 500k (5 BTC)": (450000, 550000),
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

    # --- Visualization ---
    print("\nGenerating visualizations...")

    plt.style.use('seaborn-darkgrid')
    sns.set_palette('viridis')

    # Histogram of notional quantities with logarithmic x-axis
    plt.figure(figsize=(12, 6))
    positive_quantities = df[df['notional_quantity'] > 0]['notional_quantity']
    if not positive_quantities.empty:
        sns.histplot(positive_quantities, bins=50, kde=True, log_scale=True)
        plt.xscale('log')
        plt.title(
            f'Distribution of Historical Notional Trade Quantities (Bitcoin, Scaled by {NOTIONAL_SCALING_FACTOR}, Log Scale)')
        plt.xlabel('Notional Quantity (Log Scale)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        print("  No positive quantities to plot histogram on log scale.")

    # Bar chart of categorized quantities
    plt.figure(figsize=(14, 7))
    sns.barplot(x=category_df.index, y=category_df['Percentage'], palette='coolwarm')
    plt.title(
        f'Percentage Distribution of Notional Trade Quantities by Category (Bitcoin, Scaled by {NOTIONAL_SCALING_FACTOR})')
    plt.xlabel('Notional Quantity Category')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete. Use these insights to refine your OrderSizeModel.")


# --- Main Execution ---
if __name__ == "__main__":
    historical_trades_df = load_historical_data(HISTORICAL_DATA_PATH, CSV_FILE_IN_ZIP, QUANTITY_COLUMN_INDEX,
                                                QUANTITY_COLUMN_NAME)

    if not historical_trades_df.empty:
        analyze_quantities(historical_trades_df, QUANTITY_COLUMN_NAME)

