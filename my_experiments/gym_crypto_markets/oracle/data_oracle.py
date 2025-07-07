import datetime as dt
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from abides_core import NanosecondTime

from abides_markets.oracles import Oracle

logger = logging.getLogger(__name__)


class DataOracle(Oracle):
    """
    The DataOracle provides price information to agents from a historical dataset.
    It requires a CSV file with at least 'Timestamp' and 'Price' columns.
    When an agent queries the oracle at a specific time, it returns the price
    from the historical data that corresponds to that timestamp.
    """

    def __init__(
            self,
            mkt_open: NanosecondTime,
            mkt_close: NanosecondTime,
            symbols: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Initializes the DataOracle.

        Args:
            mkt_open: The market open time.
            mkt_close: The market close time.
            symbols: A dictionary where keys are symbol names and values are
                     dictionaries containing the 'data_file' path.
        """
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close
        self.symbols: Dict[str, Dict[str, Any]] = symbols

        # The dictionary price_data holds the historical price series for each symbol.
        self.price_data: Dict[str, pd.Series] = {}

        then = dt.datetime.now()

        for symbol, params in symbols.items():
            logger.debug(f"DataOracle loading historical data for {symbol}")
            try:
                self.price_data[symbol] = self.load_historical_data(params['data_file'])
            except FileNotFoundError:
                logger.error(f"Data file not found for symbol {symbol} at path: {params['data_file']}")
                raise
            except KeyError:
                logger.error(f"The 'data_file' key was not found in the symbols configuration for {symbol}.")
                raise

        now = dt.datetime.now()

        logger.debug(f"DataOracle initialized for symbols {self.symbols.keys()}")
        logger.debug(f"DataOracle initialization took {now - then}")

    def load_historical_data(self, file_path: str) -> pd.Series:
        """
        Loads historical price data from a CSV file.

        The CSV file must have 'Timestamp' and 'Price' columns. The 'Timestamp'
        column will be set as the index.

        Args:
            file_path: The path to the CSV file.

        Returns:
            A pandas Series with timestamps as the index and prices as the values.
        """
        df = pd.read_csv(file_path, index_col='TIMESTAMP', parse_dates=True)
        if 'PRICE' not in df.columns:
            raise ValueError("The data file must contain a 'Price' column.")

        return (df['PRICE']).astype(int)

    def get_daily_open_price(
            self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """
        Return the daily open price for the given symbol.

        This will be the first price at or after the market open time.
        """

        if symbol not in self.price_data:
            raise ValueError(f"Unknown symbol: {symbol}")

        # Find the price at the market open time.
        try:
            query_time_ns = pd.to_datetime(mkt_open)

            # Floor to nearest second (last available price at or before)
            query_time_sec = query_time_ns.floor('S')
            open_price = self.price_data[symbol].asof(query_time_sec)

        except KeyError:
            logger.warning(f"No opening price found for {symbol} at {mkt_open}. Returning first available price.")
            open_price = self.price_data[symbol].iloc[0]
        #
        # if pd.isna(open_price):
        #     open_price = self.price_data[symbol].iloc[0]  # fallback to first available

        logger.debug(f"Oracle: client requested {symbol} at market open: {mkt_open}")
        logger.debug(f"Oracle: market open price was {open_price}")

        return open_price

    def observe_price(
        self,
        symbol: str,
        current_time: int,  # nanosecond timestamp (numpy.int64)
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int:
        price_series = self.price_data[symbol]

        if current_time < self.mkt_open:
            logger.warning(
                f"Oracle: attempted to observe {symbol} before start_time ({current_time} < {self.mkt_open})")
            return self.get_daily_open_price(symbol)

        # Convert current_time to pd.Timestamp for comparison
        current_ts = pd.Timestamp(current_time)

        if current_ts < price_series.index[0]:
            price_at_time = price_series.iloc[0]
            logger.warning(f"Requested time {current_ts} before first price; using first price {price_at_time}")
        elif current_ts > price_series.index[-1]:
            price_at_time = price_series.iloc[-1]
            logger.warning(f"Requested time {current_ts} after last price; using last price {price_at_time}")
        else:
            price_at_time = price_series.asof(current_ts)
            if pd.isna(price_at_time):
                logger.warning(f"No price found at {current_ts} via asof; using last known price")
                price_at_time = price_series.iloc[-1]

        if sigma_n == 0:
            observed_price = int(price_at_time)
        else:
            observed_price = int(round(random_state.normal(loc=price_at_time, scale=np.sqrt(sigma_n))))

        logger.debug(f"Oracle: observed price for {symbol} at {current_ts} is {observed_price}")
        return observed_price
