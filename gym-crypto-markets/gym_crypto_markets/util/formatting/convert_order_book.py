import argparse
import os
import pandas as pd
import numpy as np

import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[2])  # directory two levels up from this file
sys.path.append(p)

from ...util.formatting.convert_order_stream import get_year_month_day, get_start_end_time, dir_path, check_positive
from tqdm import tqdm


def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        if isinstance(x, (list, tuple)):
            for y in flatten(x):
                yield y
        else:
            yield x

def get_book_df(processed_book_levels_df, quote_levels):
    """Returns a dataframe of the orderbook from a list of dicts of the orderbook."""
    cols = []
    for i in range(1, quote_levels + 1):
        cols.append("ask_price_{}".format(i))
        cols.append("ask_size_{}".format(i))
    for i in range(1, quote_levels + 1):
        cols.append("bid_price_{}".format(i))
        cols.append("bid_size_{}".format(i))
    book_df = pd.DataFrame(processed_book_levels_df, columns=cols)
    return book_df

def get_larger_int_and_gap(a, b):
    return (True, a - b) if a >= b else (False, b - a)


def get_int_from_string(s):
    int_list = [int(s) for s in s.split('_') if s.isdigit()]
    return int_list[0]


def process_row(row_series, quote_levels):
    """
    Processes a single row (as a pandas Series) of the orderbook.
    This version directly accesses 'bids' and 'asks' lists.
    """
    row_dict = {}

    # Directly get the 'bids' and 'asks' lists from the Series.
    # Use .get() with a default empty list to handle cases where one might be missing.
    bids = row_series.get('bids', [])
    asks = row_series.get('asks', [])

    # Sort bids descending by price (best bid is highest price)
    # and asks ascending by price (best ask is lowest price).
    bids = sorted(bids, key=lambda x: x[0], reverse=True)
    asks = sorted(asks, key=lambda x: x[0])

    # Populate the dictionary with ask levels
    for i in range(1, quote_levels + 1):
        if i <= len(asks):
            price, size = asks[i - 1]
            row_dict[f"ask_price_{i}"] = price
            row_dict[f"ask_size_{i}"] = size
        else:
            row_dict[f"ask_price_{i}"] = np.nan
            row_dict[f"ask_size_{i}"] = np.nan

    # Populate the dictionary with bid levels
    for i in range(1, quote_levels + 1):
        if i <= len(bids):
            price, size = bids[i - 1]
            row_dict[f"bid_price_{i}"] = price
            row_dict[f"bid_size_{i}"] = size
        else:
            row_dict[f"bid_price_{i}"] = np.nan
            row_dict[f"bid_size_{i}"] = np.nan

    return row_dict


def reorder_columns(unordered_cols):
    """ Reorders column list to coincide with columns of LOBSTER csv file format. """

    ask_price_cols = [label for label in unordered_cols if 'ask_price' in label]
    ask_size_cols = [label for label in unordered_cols if 'ask_size' in label]
    bid_price_cols = [label for label in unordered_cols if 'bid_price' in label]
    bid_size_cols = [label for label in unordered_cols if 'bid_size' in label]

    bid_price_cols.sort(key=get_int_from_string)
    bid_size_cols.sort(key=get_int_from_string)
    ask_price_cols.sort(key=get_int_from_string)
    ask_size_cols.sort(key=get_int_from_string)

    bid_price_cols = np.array(bid_price_cols)
    bid_size_cols = np.array(bid_size_cols)
    ask_price_cols = np.array(ask_price_cols)
    ask_size_cols = np.array(ask_size_cols)

    new_col_list_size = ask_price_cols.size + ask_size_cols.size + bid_price_cols.size + bid_size_cols.size
    new_col_list = np.empty((new_col_list_size,), dtype='<U16')

    new_col_list[0::4] = ask_price_cols
    new_col_list[1::4] = ask_size_cols
    new_col_list[2::4] = bid_price_cols
    new_col_list[3::4] = bid_size_cols
    return new_col_list


def finalise_processing(orderbook_df, level):
    """ Clip to requested level and fill NaNs according to LOBSTER spec. """

    bid_columns = [col for col in orderbook_df.columns if "bid" in col]
    ask_columns = [col for col in orderbook_df.columns if "ask" in col]

    orderbook_df[bid_columns] = orderbook_df[bid_columns].fillna(value=-9999999999)
    orderbook_df[ask_columns] = orderbook_df[ask_columns].fillna(value=9999999999)

    num_levels = int((len(orderbook_df.columns) - 1) / 4) + 1
    columns_to_drop = []
    columns_to_drop.extend([f'ask_price_{idx}' for idx in range(1, num_levels + 1) if idx > level])
    columns_to_drop.extend([f'ask_size_{idx}' for idx in range(1, num_levels + 1) if idx > level])
    columns_to_drop.extend([f'bid_price_{idx}' for idx in range(1, num_levels + 1) if idx > level])
    columns_to_drop.extend([f'bid_size_{idx}' for idx in range(1, num_levels + 1) if idx > level])

    orderbook_df = orderbook_df.drop(columns=columns_to_drop)

    return orderbook_df


def is_wide_book(df):
    """ Checks if orderbook dataframe is in wide or skinny format. """
    if isinstance(df.index, pd.MultiIndex):
        return False
    else:
        return True


def process_orderbook(df, num_levels):
    """Returns a dataframe of the orderbook."""
    processed_book_levels = []

    # This loop correctly unpacks the tuple from iterrows() into 'index' and 'row_series'.
    if not is_wide_book(df):  # skinny format (MultiIndex)
        unique_timestamps = df.index.get_level_values(0).unique()
        for ts in tqdm(unique_timestamps, desc="Processing order book"):
            row_series = df.loc[ts]
            # When grouped, the series might have a single 'bids'/'asks' row.
            # We need to handle the structure properly.
            if isinstance(row_series, pd.DataFrame):
                # Take the first row if multiple events happened at the exact same ns
                row_series = row_series.iloc[0]
            row_dict = process_row(row_series, num_levels)
            processed_book_levels.append(row_dict)

        book_df = get_book_df(processed_book_levels, num_levels)
        book_df.index = unique_timestamps

    else:  # wide format (single index)
        for index, row_series in tqdm(df.iterrows(), total=len(df), desc="Processing order book"):
            # We pass only the row_series (the data) to the processing function.
            row_dict = process_row(row_series, num_levels)
            processed_book_levels.append(row_dict)

        book_df = get_book_df(processed_book_levels, num_levels)
        book_df.index = df.index

    return book_df


def save_formatted_order_book(orderbook_bz2, ticker, level, out_dir='.'):
    """ Saves orderbook data from ABIDES in LOBSTER format.

        :param orderbook_bz2: file path of order book bz2 output file.
        :type orderbook_bz2: str
        :param ticker: label of security
        :type ticker: str
        :param level: maximum level of order book to display
        :type level: int
        :param out_dir: path to output directory
        :type out_dir: str

        :return:

        ============

        Orderbook File:     (Matrix of size: (Nx(4xNumberOfLevels)))
        ---------------

        Name:   TICKER_Year-Month-Day_StartTime_EndTime_orderbook_LEVEL.csv

        Columns:

            1.) Ask Price 1:    Level 1 Ask Price   (Best Ask)
            2.) Ask Size 1:     Level 1 Ask Volume  (Best Ask Volume)
            3.) Bid Price 1:    Level 1 Bid Price   (Best Bid)
            4.) Bid Size 1:     Level 1 Bid Volume  (Best Bid Volume)
            5.) Ask Price 2:    Level 2 Ask Price   (2nd Best Ask)
            ...


    """

    orderbook_df = pd.read_pickle(orderbook_bz2)

    if not is_wide_book(orderbook_df):  # skinny format
        trading_day = get_year_month_day(pd.Series(orderbook_df.index.levels[0]))
        start_time, end_time = get_start_end_time(orderbook_df, 'orderbook_skinny')
    else:  # wide format
        trading_day = get_year_month_day(pd.Series(orderbook_df.index))
        start_time, end_time = get_start_end_time(orderbook_df, 'orderbook_wide')

    orderbook_df = process_orderbook(orderbook_df, level)

    # Save to file

    #filename = f'{ticker}_{trading_day}_{start_time}_{end_time}_orderbook_{str(level)}.csv'
    filename = f'orderbook.csv'
    filename = os.path.join(out_dir, filename)

    orderbook_df.to_csv(filename, index=False, header=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process ABIDES order book data into the LOBSTER format.')
    parser.add_argument('book', type=str, help='ABIDES order book in bz2 format. '
                                               'Typical example is `orderbook_TICKER.bz2`')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)
    parser.add_argument('ticker', type=str, help="Ticker label")
    parser.add_argument('level', type=check_positive, help="Maximum orderbook level.")

    args, remaining_args = parser.parse_known_args()

    save_formatted_order_book(args.book, args.ticker, args.level, out_dir=args.output_dir)
