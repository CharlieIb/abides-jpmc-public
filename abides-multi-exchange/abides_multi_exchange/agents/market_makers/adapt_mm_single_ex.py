import logging
from math import floor, ceil
from typing import Dict, List, Optional, Tuple

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.utils import sigmoid
from abides_markets.messages.marketdata import (
    MarketDataMsg,
    L2SubReqMsg,
    BookImbalanceDataMsg,
    BookImbalanceSubReqMsg,
    MarketDataEventMsg,
)
from abides_markets.messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from abides_markets.orders import Side
from ..trading_agent import TradingAgent

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

    ADAPTIVE SPREAD: the market maker's spread can be set either as a fixed value or can be adaptive.

    ME update: This market maker now operates on a single, assigned exchange.
    It directs all its quoting and data request activities to this specific exchange.
    """

    def __init__(
            self,
            id: int,
            symbol: str,
            starting_cash: int,
            exchange_id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            pov: float = 0.05,
            min_order_size: int = 20,
            window_size: float = 5,
            anchor: str = ANCHOR_MIDDLE_STR,
            num_ticks: int = 20,
            level_spacing: float = 0.5,
            wake_up_freq: str = "1s",
            poisson_arrival: bool = True,
            subscribe: bool = False,
            subscribe_freq: int = 10_000_000_000,
            subscribe_num_levels: int = 1,
            cancel_limit_delay: int = 50,
            skew_beta=0,
            price_skew_param=None,
            spread_alpha: float = 0.85,
            backstop_quantity: int = 0,
            log_orders: bool = False,
            min_imbalance=0.9,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders, exchange_id=exchange_id)
        self.exchange_id = exchange_id
        self.is_adaptive: bool = False
        self.symbol: str = symbol
        self.pov: float = pov
        self.min_order_size: int = min_order_size
        self.anchor: str = self.validate_anchor(anchor)
        self.window_size: float = self.validate_window_size(window_size)
        self.num_ticks: int = num_ticks
        self.level_spacing: float = level_spacing
        self.wake_up_freq: str = wake_up_freq
        self.poisson_arrival: bool = poisson_arrival
        if self.poisson_arrival:
            self.arrival_rate = str_to_ns(self.wake_up_freq)

        self.subscribe: bool = subscribe
        self.subscribe_freq: int = subscribe_freq
        self.min_imbalance = min_imbalance
        self.subscribe_num_levels: int = subscribe_num_levels
        self.cancel_limit_delay: int = cancel_limit_delay

        self.skew_beta = skew_beta
        self.price_skew_param = price_skew_param
        self.spread_alpha: float = spread_alpha
        self.backstop_quantity: int = backstop_quantity
        self.log_orders: float = log_orders

        self.has_subscribed_imbalance = False

        ## Internal variables
        self.subscription_requested: bool = False
        self.state: Dict[str, bool] = self.initialise_state()
        self.buy_order_size: int = self.min_order_size
        self.sell_order_size: int = self.min_order_size

        self.last_mid: Optional[int] = None
        self.last_spread: float = INITIAL_SPREAD_VALUE
        self.tick_size: Optional[int] = (
            None if self.is_adaptive else ceil(self.last_spread * self.level_spacing)
        )
        self.LIQUIDITY_DROPOUT_WARNING: str = (
            f"Liquidity dropout for agent {self.name}."
        )

    def initialise_state(self) -> Dict[str, bool]:
        """Returns variables that keep track of whether spread and transacted volume have been observed."""
        if self.subscribe:
            return {"AWAITING_MARKET_DATA": True, "AWAITING_TRANSACTED_VOLUME": True}
        else:
            return {"AWAITING_SPREAD": True, "AWAITING_TRANSACTED_VOLUME": True}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime):
        """Agent wakeup is determined by self.wake_up_freq."""
        can_trade = super().wakeup(current_time)

        if not self.has_subscribed_imbalance:
            super().request_data_subscription(
                self.exchange_id,
                BookImbalanceSubReqMsg(
                    symbol=self.symbol,
                    min_imbalance=self.min_imbalance,
                )
            )
            self.has_subscribed_imbalance = True

        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                self.exchange_id,
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=self.subscribe_freq,
                    depth=self.subscribe_num_levels,
                )
            )
            self.subscription_requested = True
            self.get_transacted_volume(self.exchange_id, self.symbol, lookback_period=self.subscribe_freq)
            self.state = self.initialise_state()

        elif can_trade and not self.subscribe:
            self.cancel_all_orders()
            self.delay(self.cancel_limit_delay)
            self.get_current_spread(self.exchange_id, self.symbol, depth=self.subscribe_num_levels)
            self.get_transacted_volume(self.exchange_id, self.symbol, lookback_period=self.wake_up_freq)
            self.initialise_state()

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Processes message from the assigned exchange."""
        super().receive_message(current_time, sender_id, message)

        # Only process messages from the assigned exchange
        if sender_id != self.exchange_id:
            return

        mid = self.last_mid if self.last_mid is not None else None

        if self.last_spread is not None and self.is_adaptive:
            self._adaptive_update_window_and_tick_size()

        if (
                isinstance(message, QueryTransactedVolResponseMsg)
                and self.state["AWAITING_TRANSACTED_VOLUME"]
        ):
            self.update_order_size()
            self.state["AWAITING_TRANSACTED_VOLUME"] = False

        if isinstance(message, BookImbalanceDataMsg):
            if message.stage == MarketDataEventMsg.Stage.START and mid is not None:
                self.place_orders(self.exchange_id, mid)

        if not self.subscribe:
            if (
                    isinstance(message, QuerySpreadResponseMsg)
                    and self.state["AWAITING_SPREAD"]
            ):
                bid, _, ask, _ = self.get_known_bid_ask(self.exchange_id, self.symbol)
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = int(ask - bid)
                        self._adaptive_update_spread(spread)
                else:
                    logger.debug(f"SPREAD MISSING on exchange {self.exchange_id} at time {current_time}")

                self.state["AWAITING_SPREAD"] = False

            if (
                    not self.state["AWAITING_SPREAD"]
                    and not self.state["AWAITING_TRANSACTED_VOLUME"]
                    and mid is not None
            ):
                self.place_orders(self.exchange_id, mid)
                self.state = self.initialise_state()
                self.set_wakeup(current_time + self.get_wake_frequency())

        else:  # subscription mode
            if (
                    isinstance(message, MarketDataMsg)
                    and self.state["AWAITING_MARKET_DATA"]
            ):
                bid, _, ask, _ = self.get_known_bid_ask(self.exchange_id, self.symbol)
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = int(ask - bid)
                        self._adaptive_update_spread(spread)
                else:
                    logger.debug(f"SPREAD MISSING on exchange {self.exchange_id} at time {current_time}")

                self.state["AWAITING_MARKET_DATA"] = False

            if (
                    not self.state["AWAITING_MARKET_DATA"]
                    and not self.state["AWAITING_TRANSACTED_VOLUME"]
                    and mid is not None
            ):
                self.place_orders(self.exchange_id, mid)
                self.state = self.initialise_state()

    def _adaptive_update_spread(self, spread: int) -> None:
        """Update internal spread estimate with an exponentially weighted moving average."""
        spread_ewma = (
                self.spread_alpha * spread + (1 - self.spread_alpha) * self.last_spread
        )
        self.window_size = spread_ewma
        self.last_spread = spread_ewma

    def _adaptive_update_window_and_tick_size(self) -> None:
        """Update window size and tick size relative to internal spread estimate."""
        self.window_size = self.last_spread
        self.tick_size = round(self.level_spacing * self.window_size)
        if self.tick_size == 0:
            self.tick_size = 1

    def update_order_size(self) -> None:
        """Updates size of order to be placed based on recent transacted volume."""
        buy_transacted_volume, sell_transacted_volume = self.transacted_volume.get(self.exchange_id, {}).get(
            self.symbol, (0, 0))
        total_transacted_volume = buy_transacted_volume + sell_transacted_volume

        qty = round(self.pov * total_transacted_volume)

        if self.skew_beta == 0:
            self.buy_order_size = qty if qty >= self.min_order_size else self.min_order_size
            self.sell_order_size = qty if qty >= self.min_order_size else self.min_order_size
        else:
            holdings = self.get_holdings(self.symbol)
            proportion_sell = sigmoid(holdings, self.skew_beta)
            sell_size = ceil(proportion_sell * qty)
            buy_size = floor((1 - proportion_sell) * qty)

            self.buy_order_size = buy_size if buy_size >= self.min_order_size else self.min_order_size
            self.sell_order_size = sell_size if sell_size >= self.min_order_size else self.min_order_size

    def compute_orders_to_place(self, exchange_id: int, mid: int) -> Tuple[List[int], List[int]]:
        """Given a mid price, computes the ladder of prices for new orders."""
        mid_point = mid
        if self.price_skew_param is not None:
            buy_vol, sell_vol = self.transacted_volume.get(exchange_id, {}).get(self.symbol, (0, 0))
            if buy_vol + sell_vol > 0:
                trade_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                mid_point = int(mid + (trade_imbalance * self.price_skew_param))

        if self.anchor == ANCHOR_MIDDLE_STR:
            highest_bid = int(mid_point - floor(0.5 * self.window_size))
            lowest_ask = int(mid_point + ceil(0.5 * self.window_size))
        elif self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid_point - 1)
            lowest_ask = int(mid_point + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid_point - self.window_size)
            lowest_ask = int(mid_point + 1)
        else:  # Should not happen due to validation
            highest_bid, lowest_ask = 0, 0

        lowest_bid = highest_bid - ((self.num_ticks - 1) * self.tick_size)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * self.tick_size)

        # Filter out negative or zero prices
        bids_to_place = [p for p in range(lowest_bid, highest_bid + self.tick_size, self.tick_size) if p > 0]
        asks_to_place = [p for p in range(lowest_ask, highest_ask + self.tick_size, self.tick_size) if p > 0]

        return bids_to_place, asks_to_place

    def place_orders(self, exchange_id: int, mid: int) -> None:
        """Computes and places a ladder of orders on the assigned exchange."""
        bid_orders, ask_orders = self.compute_orders_to_place(exchange_id, mid)
        orders = []

        if self.backstop_quantity > 0 and bid_orders and ask_orders:
            # Place backstop orders at the edges of the ladder
            orders.append(
                self.create_limit_order(exchange_id, self.symbol, self.backstop_quantity, Side.BID, bid_orders[0]))
            bid_orders = bid_orders[1:]
            orders.append(
                self.create_limit_order(exchange_id, self.symbol, self.backstop_quantity, Side.ASK, ask_orders[-1]))
            ask_orders = ask_orders[:-1]

        for bid_price in bid_orders:
            orders.append(self.create_limit_order(exchange_id, self.symbol, self.buy_order_size, Side.BID, bid_price))
        for ask_price in ask_orders:
            orders.append(self.create_limit_order(exchange_id, self.symbol, self.sell_order_size, Side.ASK, ask_price))

        # Filter out None values in case create_limit_order fails a risk check
        valid_orders = [order for order in orders if order is not None]
        if valid_orders:
            self.place_multiple_orders(exchange_id, valid_orders)

    def validate_anchor(self, anchor: str) -> str:
        """Checks that the input parameter anchor takes an allowed value."""
        if anchor not in [ANCHOR_TOP_STR, ANCHOR_BOTTOM_STR, ANCHOR_MIDDLE_STR]:
            raise ValueError(
                f"Variable anchor must take the value `{ANCHOR_BOTTOM_STR}`, `{ANCHOR_MIDDLE_STR}` or `{ANCHOR_TOP_STR}`"
            )
        return anchor

    def validate_window_size(self, window_size: float) -> Optional[int]:
        """Checks that the input parameter window_size takes an allowed value."""
        try:
            return int(window_size)
        except ValueError:
            if str(window_size).lower() == "adaptive":
                self.is_adaptive = True
                self.anchor = ANCHOR_MIDDLE_STR
                return None
            else:
                raise ValueError(
                    f"Variable window_size must be of type int or string '{ADAPTIVE_SPREAD_STR}'."
                )

    def get_wake_frequency(self) -> NanosecondTime:
        """Returns the next wake-up frequency, determined by a fixed or stochastic interval."""
        if not self.poisson_arrival:
            return str_to_ns(self.wake_up_freq)
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))