import logging
from math import floor, ceil
from typing import Dict, List, Optional, Tuple

import numpy as np

from abides_core import Message, NanosecondTime

from abides_markets.utils import sigmoid
from abides_markets.messages.marketdata import (
    MarketDataMsg,
    L2SubReqMsg,
    BookImbalanceSubReqMsg,
    BookImbalanceDataMsg
)
from abides_markets.messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent

ANCHOR_TOP_STR = "top"
ANCHOR_BOTTOM_STR = "bottom"
ANCHOR_MIDDLE_STR = "middle"

ADAPTIVE_SPREAD_STR = "adaptive"
INITIAL_SPREAD_VALUE = 50

logger = logging.getLogger(__name__)


class AdaptiveMarketMakerAgent(TradingAgent):
    """
    This class implements a modification of the Chakraborty-Kearns `ladder` market-making strategy, wherein the
    the size of order placed at each level is set as a fraction of measured transacted volume in the previous time
    period.

    Can skew orders to size of current inventory using beta parameter, whence beta == 0 represents inventory being
    ignored and beta == infinity represents all liquidity placed on one side of book.

    ADAPTIVE SPREAD: the market maker's spread can be set either as a fixed or value or can be adaptive,

    ME update: This market maker now maintains an independent ladder of orders on each
    exchange it is connected to. It tracks the spread, volume, and its own
    inventory separately for each market to inform its quoting strategy.
    """

    def __init__(
            self,
            id: int,
            symbol: str,
            starting_cash: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            pov: float = 0.05,
            min_order_size: int = 20,
            window_size: float = 5,
            anchor: str = ANCHOR_MIDDLE_STR,
            num_ticks: int = 20,
            level_spacing: float = 0.5,
            wake_up_freq: NanosecondTime = 1_000_000_000,  # 1 second
            poisson_arrival: bool = True,
            subscribe: bool = False,
            subscribe_freq: float = 10e9,
            subscribe_num_levels: int = 1,
            cancel_limit_delay: int = 50,
            skew_beta=0,
            price_skew_param=None,
            spread_alpha: float = 0.85,
            backstop_quantity: int = 0,
            log_orders: bool = False,
            min_imbalance=0.9,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.is_adaptive: bool = False
        self.symbol: str = symbol
        self.pov: float = pov # fraction of transacted volume placed at each price level
        self.min_order_size: int = min_order_size # minimum size order to place at each level, if pov <= min
        self.anchor: str = self.validate_anchor(anchor) # anchor either top of window or bottom of window to mid-price
        self.window_size_init: float = self.validate_window_size(window_size) # Size in ticks (cents) of how wide the window around mid price is.
        # if equal to string 'adaptive' then ladder starts at best bid and ask
        self.num_ticks: int = num_ticks # number of ticks on each side of the window in which to place liquidity
        self.level_spacing: float = level_spacing # level spacing as a fraction of the spread
        self.wake_up_freq: str = wake_up_freq # frequency of agent wake up
        self.poisson_arrival: bool = poisson_arrival # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe: bool = subscribe # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscribe_freq: float = subscribe_freq # Frequency in nanoseconds^-1 at which to receive market udpates
        self.min_imbalance = min_imbalance
        # in subscribe mode
        self.subscribe_num_levels: int = subscribe_num_levels # Number of orderbook levels in subscription mode
        self.cancel_limit_delay: int = cancel_limit_delay # delay in nanoseconds between order cancellation and new
        # limit order calculations

        self.skew_beta = skew_beta # parameter for determining order placement imbalance
        self.price_skew_param = price_skew_param # parameter determining how much to skew price level
        self.spread_alpha: float = spread_alpha # parameter for exponentially weighted moving average
        self.backstop_quantity: int = backstop_quantity # how many orders to place at outside order level,
        # to prevent liquidity dropouts. If None then place same as at other levels
        self.log_orders: float = log_orders

        self.is_adaptive: bool = window_size == ADAPTIVE_SPREAD_STR

        self.has_subscribed: bool = False

        ## Internal variables
        self.state: Dict[int, Dict[str, bool]] = {}
        self.buy_order_size: Dict[int, int] = {}
        self.sell_order_size: Dict[int, int] = {}
        self.last_mid: Dict[int, int] = {}
        self.last_spread: Dict[int, float] = {}
        self.tick_size: Dict[int, int] = {}
        self.window_size: Dict[int, int] = {}

        self.subscription_requested: bool = False

        self.LIQUIDITY_DROPOUT_WARNING: str = (
            f"Liquidity dropout for agent {self.name}."
        )

    def initialise_state(self, exchange_id: int) -> None:
        """Initialises the state dictionary for a given exchange."""
        if self.subscribe:
            self.state[exchange_id] = {"AWAITING_MARKET_DATA": True, "AWAITING_TRANSACTED_VOLUME": True}
        else:
            self.state[exchange_id] = {"AWAITING_SPREAD": True, "AWAITING_TRANSACTED_VOLUME": True}


    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        # Initialize state for each exchange
        for ex_id in self.exchange_ids:
            self.initialise_state(ex_id)
            self.buy_order_size[ex_id] = self.min_order_size
            self.sell_order_size[ex_id] = self.min_order_size
            self.last_spread[ex_id] = INITIAL_SPREAD_VALUE
            self.window_size[ex_id] = self.window_size_init if not self.is_adaptive else INITIAL_SPREAD_VALUE
            self.tick_size[ex_id] = ceil(self.last_spread[ex_id] * self.level_spacing) if not self.is_adaptive else 1

    def kernel_stopping(self) -> None:
        """The parent TradingAgent class now handles all final valuation."""
        super().kernel_stopping()


    def wakeup(self, current_time: NanosecondTime):
        """Agent wakeup is determined by self.wake_up_freq."""
        can_trade = super().wakeup(current_time)
        if not can_trade:
            return

        if not self.has_subscribed:
            for ex_id in self.exchange_ids:
                self.request_data_subscription(
                    ex_id,
                    BookImbalanceSubReqMsg(
                        symbol=self.symbol,
                        min_imbalance=self.min_imbalance
                    )
                )
            self.has_subscribed = True

        self.cancel_all_orders()
        self.delay(self.cancel_limit_delay)

        for ex_id in self.exchange_ids:
            self.initialise_state(ex_id)  # Reset state for all exchanges
            if self.subscribe and not self.subscription_requested:
                self.request_data_subscription(ex_id, L2SubReqMsg(symbol=self.symbol, freq=self.subscribe_freq,
                                                                  depth=self.subscribe_num_levels))
            elif not self.subscribe:
                self.get_current_spread(ex_id, self.symbol, depth=self.subscribe_num_levels)

            self.get_transacted_volume(ex_id, self.symbol, lookback_period=self.wake_up_freq)

        if self.subscribe and not self.subscription_requested:
            self.subscription_requested = True

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        """Processes message from an exchange and acts on a per-exchange basis."""
        super().receive_message(current_time, sender_id, message)

        # --- MODIFICATION: Handle messages on a per-exchange (sender_id) basis ---
        exchange_state = self.state.get(sender_id)
        if not exchange_state:
            return

        if isinstance(message, BookImbalanceDataMsg):
            # This message is a trigger to requote the market.
            # We can use the last known mid-price for this exchange to place orders.
            last_mid_for_exchange = self.last_mid.get(sender_id)
            if last_mid_for_exchange:
                self.cancel_all_orders()  # First cancel existing quotes
                self.delay(self.cancel_limit_delay)
                self.place_orders(sender_id, last_mid_for_exchange)
            return

        # Handle volume responses
        if isinstance(message, QueryTransactedVolResponseMsg) and exchange_state["AWAITING_TRANSACTED_VOLUME"]:
            self.update_order_size(sender_id)
            exchange_state["AWAITING_TRANSACTED_VOLUME"] = False

        # Handle spread/market data responses
        mid = None
        if not self.subscribe and isinstance(message, QuerySpreadResponseMsg) and exchange_state["AWAITING_SPREAD"]:
            bid, _, ask, _ = self.get_known_bid_ask(sender_id, self.symbol)
            if bid and ask:
                mid = int((ask + bid) / 2)
                self.last_mid[sender_id] = mid
                if self.is_adaptive:
                    self._adaptive_update_spread(sender_id, int(ask - bid))
            exchange_state["AWAITING_SPREAD"] = False

        elif self.subscribe and isinstance(message, MarketDataMsg) and exchange_state["AWAITING_MARKET_DATA"]:
            bids = self.known_bids.get(sender_id, {}).get(self.symbol, [])
            asks = self.known_asks.get(sender_id, {}).get(self.symbol, [])
            if bids and asks:
                mid = int((bids[0][0] + asks[0][0]) / 2)
                self.last_mid[sender_id] = mid
                if self.is_adaptive:
                    self._adaptive_update_spread(sender_id, int(asks[0][0] - bids[0][0]))
            exchange_state["AWAITING_MARKET_DATA"] = False

        # Check if we have all necessary data for this exchange to place orders
        if not exchange_state["AWAITING_TRANSACTED_VOLUME"] and \
                not exchange_state.get("AWAITING_SPREAD", True) and \
                not exchange_state.get("AWAITING_MARKET_DATA", True):

            current_mid = self.last_mid.get(sender_id)
            if current_mid:
                self.place_orders(sender_id, current_mid)

            if not self.subscribe:
                self.set_wakeup(current_time + self.get_wake_frequency())

    def _adaptive_update_spread(self, exchange_id: int, spread: int) -> None:
        """Updates spread estimate for a specific exchange."""
        spread_ewma = self.spread_alpha * spread + (1 - self.spread_alpha) * self.last_spread[exchange_id]
        self.window_size[exchange_id] = spread_ewma
        self.last_spread[exchange_id] = spread_ewma
        self.tick_size[exchange_id] = max(1, round(self.level_spacing * spread_ewma))


    def get_holdings_by_exchange(self, symbol: str, exchange_id: int) -> int:
        """Gets holdings for a symbol on a specific exchange."""
        return self.holdings_by_exchange.get(exchange_id, {}).get(symbol, 0)

    def update_order_size(self, exchange_id: int) -> None:
        """Updates size of order to be placed on a specific exchange."""
        # --- MODIFICATION: Uses per-exchange volume and inventory ---
        buy_vol, sell_vol = self.transacted_volume.get(exchange_id, {}).get(self.symbol, (0, 0))
        total_vol = buy_vol + sell_vol
        qty = round(self.pov * total_vol)

        if self.skew_beta == 0:
            self.buy_order_size[exchange_id] = qty if qty >= self.min_order_size else self.min_order_size
            self.sell_order_size[exchange_id] = qty if qty >= self.min_order_size else self.min_order_size
        else:
            holdings_on_exchange = self.get_holdings_by_exchange(self.symbol, exchange_id)
            proportion_sell = sigmoid(holdings_on_exchange, self.skew_beta)
            sell_size = ceil(proportion_sell * qty)
            buy_size = floor((1 - proportion_sell) * qty)
            self.buy_order_size[exchange_id] = buy_size if buy_size >= self.min_order_size else self.min_order_size
            self.sell_order_size[exchange_id] = sell_size if sell_size >= self.min_order_size else self.min_order_size

    def compute_orders_to_place(self, exchange_id: int, mid: int) -> Tuple[List[int], List[int]]:
        """Computes the ladder of orders for a specific exchange."""
        if self.price_skew_param is None:
            mid_point = mid
        else:
            # Get the transacted volume for this specific exchange
            buy_vol, sell_vol = self.transacted_volume.get(exchange_id, {}).get(self.symbol, (0, 0))
            total_vol = buy_vol + sell_vol

            if buy_vol == 0 and sell_vol == 0:
                mid_point = mid
            else:
                trade_imbalance = (2 * buy_vol / total_vol) -1
                mid_point = int(mid + (trade_imbalance * self.price_skew_param))
        window = self.window_size[exchange_id]
        tick = self.tick_size[exchange_id]

        if self.anchor == ANCHOR_MIDDLE_STR:
            highest_bid = int(mid_point - floor(0.5 * window))
            lowest_ask = int(mid_point + ceil(0.5 * window))
        elif self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid_point - 1)
            lowest_ask = int(mid_point + window)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid_point - window)
            lowest_ask = int(mid_point + 1)


        lowest_bid = highest_bid - ((self.num_ticks - 1) * tick)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * tick)

        bids = [p for p in range(lowest_bid, highest_bid + tick, tick)] if tick > 0 else []
        asks = [p for p in range(lowest_ask, highest_ask + tick, tick)] if tick > 0 else []

        return bids, asks

    def place_orders(self, exchange_id: int, mid: int) -> None:
        """Places a ladder of orders on a specific exchange."""
        bid_orders, ask_orders = self.compute_orders_to_place(exchange_id, mid)
        orders = []

        if self.backstop_quantity > 0 and bid_orders and ask_orders:
            # Place a large order at the lowest bid price
            bid_price = bid_orders[0]
            logger.debug(f"{self.name}: Placing BUY limit order of size {self.backstop_quantity} @ price {bid_price}")
            orders.append(
                self.create_limit_order(exchange_id, self.symbol, self.backstop_quantity, Side.BID, bid_price))
            bid_orders = bid_orders[1:]  # Remove from list so it's not placed again

            # Place a large order at the highest ask price
            ask_price = ask_orders[-1]
            logger.debug(f"{self.name}: Placing SELL limit order of size {self.backstop_quantity} @ price {ask_price}")
            orders.append(
                self.create_limit_order(exchange_id, self.symbol, self.backstop_quantity, Side.ASK, ask_price))
            ask_orders = ask_orders[:-1]  # Remove from list

        buy_size = self.buy_order_size[exchange_id]
        sell_size = self.sell_order_size[exchange_id]

        for bid_price in bid_orders:
            logger.debug(f"{self.name}: Placing BUY limit order of size {self.buy_order_size} @ price {bid_price}")
            orders.append(self.create_limit_order(exchange_id, self.symbol, buy_size, Side.BID, bid_price))
        for ask_price in ask_orders:
            logger.debug(f"{self.name}: Placing SELL limit order of size {self.sell_order_size} @ price {ask_price}")
            orders.append(self.create_limit_order(exchange_id, self.symbol, sell_size, Side.ASK, ask_price))

        # Filter out None values in case create_limit_order fails a risk check
        valid_orders = [order for order in orders if order is not None]
        if valid_orders:
            self.place_multiple_orders(exchange_id, valid_orders)

    def validate_anchor(self, anchor: str) -> str:
        if anchor not in [ANCHOR_TOP_STR, ANCHOR_BOTTOM_STR, ANCHOR_MIDDLE_STR]:
            raise ValueError(
                f"Variable anchor must take the value `{ANCHOR_BOTTOM_STR}`, `{ANCHOR_MIDDLE_STR}` or `{ANCHOR_TOP_STR}`")
        return anchor

    def validate_window_size(self, window_size: float) -> Optional[int]:
        """Checks that input parameter window_size takes allowed value."""
        try:
            return int(window_size)
        except ValueError:
            if str(window_size).lower() == "adaptive":
                self.is_adaptive = True
                self.anchor = ANCHOR_MIDDLE_STR
                return None  # Will be set adaptively later
            else:
                raise ValueError(f"Variable window_size must be of type int or string 'adaptive'.")

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

