import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import json
import argparse

# Ensure the path to the ABIDES utilities is correct
sys.path.append('../../../../..')
from gym_crypto_markets.util.plotting.realism_utils import make_orderbook_for_analysis

# --- Configuration ---
DEPTH = 20  # How many levels of the order book to analyze
DOWNSAMPLE_FREQ = '10S'  # Resample the book every 10 seconds to make the plot manageable


def create_orderbooks_with_depth(exchange_path, ob_path, depth):
    """ Creates an orderbook DataFrame with specified depth. """
    print(f"Constructing orderbook with {depth} levels of depth...")
    # This function from realism_utils can fetch multiple levels
    processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=depth,
                                                      hide_liquidity_collapse=False)
    return processed_orderbook


def plot_market_depth_heatmap(orderbook_df, depth, config, title=None, out_file="market_depth_heatmap.png"):
    """
    Produces a heatmap of order book depth over time.

    - X-axis: Time of day
    - Y-axis: Price Level
    - Color: Volume of orders available
    """
    print("Generating market depth heatmap...")
    fig, ax = plt.subplots(figsize=(15, 10))

    # 1. Determine time and price ranges for the plot axes
    date = orderbook_df.index[0].date()
    midnight = pd.Timestamp(date)
    xmin_dt = midnight + pd.to_timedelta(config['xmin'])
    xmax_dt = midnight + pd.to_timedelta(config['xmax'])

    # Filter the dataframe to the relevant time window first
    ob_slice = orderbook_df.loc[xmin_dt:xmax_dt]

    # Find price range, ignoring outliers for a better view
    all_prices = []
    for i in range(1, depth + 1):
        all_prices.extend(ob_slice[f'bid_price_{i}'].dropna())
        all_prices.extend(ob_slice[f'ask_price_{i}'].dropna())

    if not all_prices:
        print("No order book data in the specified time range. Cannot generate plot.")
        return

    p_low_cents = np.percentile(all_prices, 0.5)
    p_high_cents = np.percentile(all_prices, 99.5)

    # 2. Create a grid for the heatmap
    # Downsample the data to make the plot less dense
    resampled_ob = ob_slice.resample(DOWNSAMPLE_FREQ).last().dropna(how='all')
    time_bins = resampled_ob.index

    # Use integer cents for price bins to avoid floating point issues
    price_bins = np.arange(int(p_low_cents), int(p_high_cents) + 1)

    bid_grid = pd.DataFrame(0, index=price_bins, columns=time_bins)
    ask_grid = pd.DataFrame(0, index=price_bins, columns=time_bins)

    # 3. Populate the grid with volume data
    for t, row in resampled_ob.iterrows():
        for i in range(1, depth + 1):
            # Populate bids
            bid_price = row.get(f'bid_price_{i}')
            if pd.notna(bid_price):
                bid_grid.loc[int(bid_price), t] = row.get(f'bid_size_{i}', 0)
            # Populate asks
            ask_price = row.get(f'ask_price_{i}')
            if pd.notna(ask_price):
                ask_grid.loc[int(ask_price), t] = row.get(f'ask_size_{i}', 0)
    print(bid_grid)
    print(ask_grid)

    bid_grid.replace(0, np.nan, inplace=True)
    ask_grid.replace(0, np.nan, inplace=True)

    all_volumes = np.concatenate([bid_grid[bid_grid > 0].values, ask_grid[ask_grid > 0].values])
    vmax = np.nanpercentile(all_volumes, 98) if len(all_volumes) > 0 else 1

    log_norm = mcolors.LogNorm(vmin=1, vmax=vmax)

    # 4. Plot the heatmaps
    # Define color maps
    bid_cmap = mcolors.LinearSegmentedColormap.from_list("bids", ["white", "blue"])
    ask_cmap = mcolors.LinearSegmentedColormap.from_list("ask", ["white", "red"])

    # Plot asks first, then bids on top with some transparency
    im_asks = ax.imshow(ask_grid, aspect='auto', cmap=ask_cmap, norm=log_norm,
                        extent=[mdates.date2num(time_bins[0]), mdates.date2num(time_bins[-1]),
                                price_bins[0] / 100, price_bins[-1] / 100],
                        origin='lower', interpolation='nearest')

    im_bids = ax.imshow(bid_grid, aspect='auto', cmap=bid_cmap, norm=log_norm, alpha=0.7,
                        extent=[mdates.date2num(time_bins[0]), mdates.date2num(time_bins[-1]),
                                price_bins[0] / 100, price_bins[-1] / 100],
                        origin='lower', interpolation='nearest')

    # 5. Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_ylabel("Price ($)", fontsize=14)
    ax.set_xlabel("Time", fontsize=14)

    # --- THIS IS THE FIX ---
    # Convert the Timestamp limits to matplotlib's numeric date format before setting.
    ax.set_xlim(mdates.date2num(xmin_dt), mdates.date2num(xmax_dt))
    # -----------------------

    ax.set_ylim(p_low_cents/100, p_high_cents/100)
    if title:
        ax.set_title(title, fontsize=16)

    ax.ticklabel_format(style='plain', axis='y')

    fig.colorbar(im_asks, ax=ax, label="Ask Volume", pad=0.01)
    fig.colorbar(im_bids, ax=ax, label="Bid Volume", pad=0.06)

    fig.savefig(out_file, format='png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {out_file}")


def main(exchange_path, ob_path, config, title=None, outfile='market_depth_heatmap.png'):
    """ Main execution function. """
    orderbook_df = create_orderbooks_with_depth(exchange_path, ob_path, depth=DEPTH)
    plot_market_depth_heatmap(orderbook_df, depth=DEPTH, config=config, title=title, out_file=outfile)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a heatmap of market depth over time from ABIDES data.')
    parser.add_argument('stream', type=str, help='ABIDES order stream file (e.g., ExchangeAgent.bz2)')
    parser.add_argument('book', type=str, help='ABIDES order book file (e.g., ORDERBOOK_TICKER_FULL.bz2)')
    parser.add_argument('-o', '--out_file', help='Path to png output file.', default='market_depth_heatmap.png')
    parser.add_argument('-t', '--plot-title', help="Title for the plot.", type=str, default="Market Depth Heatmap")
    parser.add_argument('-c', '--plot-config', help='JSON config file for plot parameters (e.g., time window).',
                        default='/home/charlie/PycharmProjects/ABIDES_GYM_EXT/abides-jpmc-public/gym-crypto-markets/gym_crypto_markets/util/plotting/configs/telemetary_config.json', type=str)

    args = parser.parse_args()

    with open(args.plot_config, 'r') as f:
        PLOT_PARAMS_DICT = json.load(f)

    main(args.stream, args.book, config=PLOT_PARAMS_DICT, title=args.plot_title, outfile=args.out_file)