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
    BookImbalanceSubReqMsg,
    BookImbalanceDataMsg
)
from abides_markets.messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from abides_markets.orders import Side
from abides_multi_exchange.agents.trading_agent import TradingAgent

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
            exchange_id: int = None,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            pov: float = 0.05,
            min_order_size: int = 20,
            window_size: float = 5,
            anchor: str = ANCHOR_MIDDLE_STR,
            num_ticks: int = 20,
            level_spacing: float = 0.5,
            wake_up_freq: str = "1s",  # 1 second
            poisson_arrival: bool = True,
            cancel_limit_delay: int = 50,
            skew_beta=0,
            price_skew_param=None,
            spread_alpha: float = 0.85,
            backstop_quantity: int = 0,
            log_orders: bool = False,
            min_imbalance=0.9,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders, exchange_id=exchange_id)
        self.exchange_id=exchange_id
        self.symbol: str = symbol
        self.pov: float = pov # fraction of transacted volume placed at each price level
        self.min_order_size: int = min_order_size # minimum size order to place at each level, if pov <= min
        self.anchor: str = self.validate_anchor(anchor) # anchor either top of window or bottom of window to mid-price
        # if equal to string 'adaptive' then ladder starts at best bid and ask
        self.num_ticks: int = num_ticks # number of ticks on each side of the window in which to place liquidity
        self.level_spacing: float = level_spacing # level spacing as a fraction of the spread
        self.wake_up_freq: str = wake_up_freq # frequency of agent wake up
        self.poisson_arrival: bool = poisson_arrival # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq
        self.cancel_limit_delay: int = cancel_limit_delay # delay in nanoseconds between order cancellation and new
        self.skew_beta = skew_beta # parameter for determining order placement imbalance
        self.price_skew_param = price_skew_param # parameter determining how much to skew price level
        self.spread_alpha: float = spread_alpha # parameter for exponentially weighted moving average
        self.backstop_quantity: int = backstop_quantity # how many orders to place at outside order level,
        # # to prevent liquidity dropouts. If None then place same as at other levels
        self.log_orders: float = log_orders
        self.min_imbalance = min_imbalance

        # self.has_subscribed: bool = False

        ## Internal variables
        self.state: Dict[str, bool] = self.initialise_state()
        self.buy_order_size: int = self.min_order_size
        self.sell_order_size: int = self.min_order_size
        self.last_mid: Optional[int] = None
        self.last_spread: float = INITIAL_SPREAD_VALUE
        self.is_adaptive: bool = False  # Will be set by validate_window_size
        self.window_size: Optional[int] = self.validate_window_size(window_size)
        self.tick_size: int = ceil(self.last_spread * self.level_spacing) if not self.is_adaptive else 1

        self.is_quoting: bool = False
        self.quoting_cycle_start_time: Optional[NanosecondTime] = None

        self.LIQUIDITY_DROPOUT_WARNING: str = (
            f"Liquidity dropout for agent {self.name}."
        )

    def initialise_state(self) -> Dict[str, bool]:
        """Initialises the state dictionary for a given exchange."""
        return {"AWAITING_SPREAD": True, "AWAITING_TRANSACTED_VOLUME": True}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        # Subscribe to the single assigned exchange
        super().request_data_subscription(
            self.exchange_id,
            BookImbalanceSubReqMsg(
                symbol=self.symbol,
                min_imbalance=self.min_imbalance,
            )
        )

    def kernel_stopping(self) -> None:
        """The parent TradingAgent class now handles all final valuation."""
        super().kernel_stopping()


    def wakeup(self, current_time: NanosecondTime):
        """Agent wakeup is determined by self.wake_up_freq."""
        # print(f"\n--- WAKEUP at {current_time} for {self.name} ---")
        can_trade = super().wakeup(current_time)
        if not can_trade:
            # print(f"DEBUG ({self.name}): Market not open or not yet known. Sleeping.")
            return
        # Could add this back in to ensure that the agent is subscribed to their exchange
        # if not self.has_subscribed:
        #     for ex_id in self.exchange_ids:
        #         self.request_data_subscription(
        #             ex_id,
        #             BookImbalanceSubReqMsg(
        #                 symbol=self.symbol,
        #                 min_imbalance=self.min_imbalance
        #             )
        #         )
        #     self.has_subscribed = True
        # print(f"DEBUG ({self.name}): Cancelling all orders to requote.")
        if self.is_quoting:
            timeout = str_to_ns("5s")  # Example: 5 second timeout
            if current_time - self.quoting_cycle_start_time > timeout:
                print(f"WARNING ({self.name}): Quoting cycle timed out. Resetting.")
                self.is_quoting = False

        if not self.is_quoting:
            self.start_quoting_cycle()

    def start_quoting_cycle(self):
        """A helper function to begin the process of fetching data to place a quote."""
        if self.is_quoting:
            return
        self.is_quoting = True

        self.quoting_cycle_start_time = self.current_time

        self.state = self.initialise_state()
        # print(
        #     f"DEBUG ({self.name} @ {self.current_time}): sending messages to Exchange {self.exchange_id}.")
        self.get_current_spread(self.exchange_id, self.symbol)
        self.get_transacted_volume(self.exchange_id, self.symbol, lookback_period=self.wake_up_freq)


    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        """Processes message from an exchange and acts on a per-exchange basis."""
        super().receive_message(current_time, sender_id, message)
        # print(
        #     f"DEBUG ({self.name} @ {current_time}): Received message '{type(message).__name__}' from Exchange {sender_id}.")
        # Ignore messages from exchanges this agent is not assigned to
        if sender_id != self.exchange_id:
            return

        # Handle the book imbalance trigger
        if isinstance(message, BookImbalanceDataMsg):
            # This message is a trigger to requote the market.
            if not self.is_quoting:
                # print(f"DEBUG ({self.name}): Imbalance alert received. Starting new quote cycle.")
                self.start_quoting_cycle()
            # else:
            #     print(f"DEBUG ({self.name}): Imbalance alert received, but already quoting. Ignoring.")
            return

        if not self.is_quoting:
            return

        # Handle transacted volume response
        if isinstance(message, QueryTransactedVolResponseMsg) and self.state["AWAITING_TRANSACTED_VOLUME"]:
            self.update_order_size()
            # print(
            #     f"DEBUG ({self.name}): Processed volume for Exchange {sender_id}. State: {self.state}")
            self.state["AWAITING_TRANSACTED_VOLUME"] = False

        # Handle spread response
        if isinstance(message, QuerySpreadResponseMsg) and self.state["AWAITING_SPREAD"]:
            bid, _, ask, _ = self.get_known_bid_ask(self.exchange_id, self.symbol)
            if bid and ask:
                self.last_mid = int((ask + bid) / 2)
                if self.is_adaptive:
                    self._adaptive_update_spread(int(ask - bid))
            self.state["AWAITING_SPREAD"] = False
            # print(
            #     f"DEBUG ({self.name}): Processed spread for Exchange {sender_id}. Last mid: {self.last_mid} State: {self.state}")

        # If all data is received, place orders
        if not self.state["AWAITING_SPREAD"] and not self.state["AWAITING_TRANSACTED_VOLUME"]:
            if self.last_mid is not None:
                # print(
                #     f"DEBUG ({self.name}): All data received for Exchange {sender_id}. Attempting to place orders.")
                self.place_orders(self.exchange_id, self.last_mid)
            # else:
            #     print(f"DEBUG ({self.name}): Cannot place orders for Exchange {sender_id}, mid-price is missing.")
            self.set_wakeup(current_time + self.get_wake_frequency())
            self.is_quoting = False




    def _adaptive_update_spread(self, spread: int) -> None:
        """Updates spread estimate for a specific exchange."""
        spread_ewma = self.spread_alpha * spread + (1 - self.spread_alpha) * self.last_spread

        # if spread > 5.0:
        #     print(f"DEBUG ({self.name}): Adaptive spread update. "
        #           f"Old Spread: {self.last_spread:.2f}, "
        #           f"Current Spread: {spread}, "
        #           f"New Spread (Window): {spread_ewma:.2f}")

        self.window_size = spread_ewma
        self.last_spread = spread_ewma
        self.tick_size = max(1, round(self.level_spacing * spread_ewma))


    def update_order_size(self) -> None:
        """Updates size of order to be placed on a specific exchange."""

        buy_vol, sell_vol = self.transacted_volume.get(self.exchange_id).get(self.symbol, (0, 0))
        total_vol = buy_vol + sell_vol
        qty = round(self.pov * total_vol)
        # print(f"DEBUG ({self.name} for Ex {self.exchange_id}): Total vol={total_vol}, base size={qty}.")

        if self.skew_beta == 0:
            self.buy_order_size = qty if qty >= self.min_order_size else self.min_order_size
            self.sell_order_size = qty if qty >= self.min_order_size else self.min_order_size
            # print(
            #     f"DEBUG ({self.name} for Ex {self.exchange_id}): Final order sizes -> BUY: {self.buy_order_size}, SELL: {self.sell_order_size}")
        else:
            holdings = self.get_holdings(self.symbol)
            proportion_sell = sigmoid(holdings, self.skew_beta)
            sell_size = ceil(proportion_sell * qty)
            buy_size = floor((1 - proportion_sell) * qty)
            self.buy_order_size = buy_size if buy_size >= self.min_order_size else self.min_order_size
            self.sell_order_size = sell_size if sell_size >= self.min_order_size else self.min_order_size
            # if self.buy_order_size > 1 or self.sell_order_size > 1:
            #     print(
            #         f"DEBUG ({self.name} for Ex {self.exchange_id}): Final order sizes -> BUY: {self.buy_order_size}, SELL: {self.sell_order_size}")
    def compute_orders_to_place(self, mid: int) -> Tuple[List[int], List[int]]:
        """Computes the ladder of orders for a specific exchange."""
        mid_point = mid
        holdings = self.get_holdings(self.symbol)


        if self.price_skew_param is not None and holdings != 0 :
            if holdings > 0:
                mid_point += floor(-1 * self.price_skew_param * np.tanh(holdings / (self.min_order_size * 10)))
            else:
                mid_point += ceil(1 * self.price_skew_param * np.tanh(abs(holdings) / (self.min_order_size * 10)))


        if self.anchor == ANCHOR_MIDDLE_STR:
            highest_bid = int(mid_point - floor(0.5 * self.window_size))
            lowest_ask = int(mid_point + ceil(0.5 * self.window_size))
        elif self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid_point - 1)
            lowest_ask = int(mid_point + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid_point - self.window_size)
            lowest_ask = int(mid_point + 1)


        lowest_bid = highest_bid - ((self.num_ticks - 1) * self.tick_size)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * self.tick_size)

        bids_raw = [p for p in range(lowest_bid, highest_bid + self.tick_size, self.tick_size)] if self.tick_size > 0 else []
        asks_raw = [p for p in range(lowest_ask, highest_ask + self.tick_size, self.tick_size)] if self.tick_size > 0 else []

        # Filter out negative priced orders
        bids = [p for p in bids_raw if p > 0]
        asks = [p for p in asks_raw if p > 0]

        # print(
        #     f"DEBUG ({self.name} for Ex {self.exchange_id}): Computed ladder -> Mid: {mid}, Bids: {bids}, Asks: {asks}")
        return bids, asks

    def place_orders(self, exchange_id: int, mid: int) -> None:
        """Places a ladder of orders on a specific exchange."""
        bid_orders, ask_orders = self.compute_orders_to_place(mid)
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


        for bid_price in bid_orders:
            logger.debug(f"{self.name}: Placing BUY limit order of size {self.buy_order_size} @ price {bid_price}")
            orders.append(self.create_limit_order(exchange_id, self.symbol, self.buy_order_size, Side.BID, bid_price))
        for ask_price in ask_orders:
            logger.debug(f"{self.name}: Placing SELL limit order of size {self.sell_order_size} @ price {ask_price}")
            orders.append(self.create_limit_order(exchange_id, self.symbol, self.sell_order_size, Side.ASK, ask_price))

        # Filter out None values in case create_limit_order fails a risk check
        valid_orders = [order for order in orders if order is not None]
        if valid_orders:
            # print(f"DEBUG ({self.name}): Placing {len(valid_orders)} orders on Exchange {exchange_id}.")
            self.place_multiple_orders(exchange_id, valid_orders)
        # else:
        #     print(f"DEBUG ({self.name}): No valid orders to place on Exchange {exchange_id}.")

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
            return str_to_ns(self.wake_up_freq)
        else:
            arrival_rate_ns = str_to_ns(self.wake_up_freq)
            delta_time = self.random_state.exponential(scale=arrival_rate_ns)
            return int(round(delta_time))

