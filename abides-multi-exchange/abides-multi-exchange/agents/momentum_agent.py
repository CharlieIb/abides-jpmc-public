from typing import List, Optional, Dict

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ...messages.marketdata import MarketDataMsg, L2SubReqMsg
from ...messages.query import QuerySpreadResponseMsg
from ...orders import Side
from ..trading_agent import TradingAgent


class MomentumAgent(TradingAgent):
    """
    Simple Trading Agent that compares the 20 past mid-price observations with the 50 past observations and places a
    buy limit order if the 20 mid-price average >= 50 mid-price average or a
    sell limit order if the 20 mid-price average < 50 mid-price average
    """

    def __init__(
        self,
        id: int,
        symbol,
        starting_cash,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        min_size=20,
        max_size=50,
        wake_up_freq: NanosecondTime = str_to_ns("60s"),
        poisson_arrival=True,
        order_size_model=None,
        subscribe=False,
        log_orders=False,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = (
            self.random_state.randint(self.min_size, self.max_size)
            if order_size_model is None
            else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list: List[float] = []
        self.avg_20_list: List[float] = []
        self.avg_50_list: List[float] = []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"

        # For polling mode
        self.latest_spreads: Dict[int, QuerySpreadResponseMsg] = {}
        self.spreads_to_receive: int = 0

        # For subscription mode, agent will pick one exchange to subscribe to
        self.subscribed_exchange: Optional[int] = None


    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def kernel_stopping(selfself) -> None:
        """
        The parent TradingAgent class now handles all final valuation
        :return:
        """
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if self.subscribe and not self.subscription_requested:
            # In subscription mode, the agent picks one exchange and subscribes to its data stream
            if self.exchange_ids:
                self.subscribed_exchange = self.random_state.choice(self.exchange_ids)
                super().request_data_subscription(
                    L2SubReqMsg(
                        symbol=self.symbol,
                        freq=int(10e9),
                        depth=1,
                    )
                )
                self.subscription_requested = True
                self.state = "AWAITING_MARKET_DATA"

        elif can_trade and not self.subscribe:
            # In polling mode, the agent queries all exchanges for their spreads
            if self.exchange_ids:
                self.spreads_to_receive = len(self.exchange_ids)
                self.latest_spreads = {} # Clear the old data
                for ex_id in self.exchange_ids:
                    self.get_current_spread(ex_id, self.symbol)
                self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Momentum agent actions are determined after obtaining the best bid and ask in the LOB"""
        super().receive_message(current_time, sender_id, message)

        # Polling Mode
        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            if len(self.latest_spreads) >= self.spreads_to_receive:
                self.analyse_and_place_order()
                self.set_wakeup(current_time + self.get_wake_frequency())
                self.state = "AWAITING_WAKEUP"

        # Subscription Mode
        elif (
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and isinstance(message, MarketDataMsg)
        ):
            bids, asks = self.known_bids.get(self.subscribed_exchange, {}).get(self.symbol, []), self.known_asks.get(self.subscribed_exchange, {}).get(self.symbol, [])
            if bids and asks:
                # Pass single exchange's data to the analysis function
                self.analyse_and_place_order(single_bid=bids[0][0], single_ask=asks[0][0], single_exchange=self.subscribed_exchange)
            self.state = "AWAITING_MARKET_DATA"

    def analyse_and_place_order(self, single_bid: int, single_ask: int, single_exchange=None) -> None:
        """
        Analyses market data to calculate momentum and places orders accordingly
        Can operate on global view (polling) or a single exchange's view (subscription)
        """
        best_global_bid, best_global_ask = float('-inf'), float('inf')
        best_bid_exchange, best_ask_exchange = None, None

        if single_exchange is not None:
            # Subscription mode: use data from single subscribed exchange
            best_global_bid, best_global_ask = single_bid, single_ask
            best_bid_exchange, best_ask_exchange = single_exchange, single_exchange
        else:
            # Polling mode: find the best bind and ask across all exchanges
            for ex_id, spread_msg in self.latest_spreads.items():
                if spread_msg.asks and spread_msg.asks[0][0] < best_global_ask:
                    best_global_ask = spread_msg.asks[0][0]
                    best_ask_exchange = ex_id
                if spread_msg.bids and spread_msg.bids[0][0] > best_global_bid:
                    best_global_bid = spread_msg.bids[0][0]
                    best_bid_exchange = ex_id


        # Proceed if we have valid spread
        if best_global_bid > float('-inf') and best_global_ask < float('inf'):
            mid = (best_global_bid + best_global_ask) / 2
            self.mid_list.append(mid)

            # Update moving averages
            if len(self.mid_list) > 20:
                self.avg_20_list.append(
                    MomentumAgent.ma(self.mid_list, n=20)[-1].round(2)
                )
            if len(self.mid_list) > 50:
                self.avg_50_list.append(
                    MomentumAgent.ma(self.mid_list, n=50)[-1].round(2)
                )
            # Check trading signal
            if len(self.avg_20_list) > 0 and len(self.avg_50_list) > 0:
                if self.order_size_model is not None:
                    self.size = self.order_size_model.sample(
                        random_state=self.random_state
                    )

                if self.size > 0:
                    # Positive momentum: buy at exchange with the best (lowest) ask
                    if self.avg_20_list[-1] >= self.avg_50_list[-1]:
                        self.place_limit_order(
                            self.symbol,
                            quantity=self.size,
                            side=Side.BID,
                            limit_price=best_global_ask,
                        )
                    # Negative momentum: sell at the exchange with best (highest) bid
                    else:
                        self.place_limit_order(
                            self.symbol,
                            quantity=self.size,
                            side=Side.ASK,
                            limit_price=best_global_bid,
                        )

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n
