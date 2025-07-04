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
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
        if 'Price' not in df.columns:
            raise ValueError("The data file must contain a 'Price' column.")

        # Ensure the prices are in integer cents.
        return (df['Price'] * 100).astype(int)

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
            open_price = self.price_data[symbol].asof(mkt_open)
        except KeyError:
            logger.warning(f"No opening price found for {symbol} at {mkt_open}. Returning first available price.")
            open_price = self.price_data[symbol].iloc[0]

        logger.debug(f"Oracle: client requested {symbol} at market open: {mkt_open}")
        logger.debug(f"Oracle: market open price was {open_price}")

        return open_price

    def observe_price(
            self,
            symbol: str,
            current_time: NanosecondTime,
            random_state: Optional[np.random.RandomState] = None,
    ) -> int:
        """
        Return the historical price at the given time.

        Args:
            symbol: The symbol to observe.
            current_time: The current simulation time.
            random_state: Not used by this oracle, maintained for compatibility.

        Returns:
            The price of the symbol at the given time.
        """
        if symbol not in self.price_data:
            raise ValueError(f"Unknown symbol: {symbol}")

        # If the request is made after market close, return the last price.
        if current_time >= self.mkt_close:
            price_at_time = self.price_data[symbol].asof(self.mkt_close - pd.Timedelta(nanoseconds=1))
        else:
            price_at_time = self.price_data[symbol].asof(current_time)

        if pd.isna(price_at_time):
            logger.warning(f"No price observation found for {symbol} at {current_time}. Returning last known price.")
            # Fallback to the last known price if no price is found at the current time.
            price_at_time = self.price_data[symbol].iloc[-1]

        logger.debug(f"Oracle: current historical price is {price_at_time} at {current_time}")

        return int(price_at_time)