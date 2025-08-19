from typing import List, Optional, Deque
from collections import deque

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.marketdata import MarketDataMsg, L2SubReqMsg
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent


class MomentumAgentSE(TradingAgent):
    """
    Simple Trading Agent that compares the 20 past mid-price observations with the 50 past observations and places a
    buy limit order if the 20 mid-price average >= 50 mid-price average or a
    sell limit order if the 20 mid-price average < 50 mid-price average.

    In a multi-exchange environment, this agent is assigned to a single exchange for its entire lifecycle.
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
            exchange_ids: Optional[List[int]] = None,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders, exchange_id=exchange_ids)
        self.symbol = symbol
        self.min_size = min_size
        self.max_size = max_size
        self.size = (
            self.random_state.randint(self.min_size, self.max_size)
            if order_size_model is None
            else None
        )
        self.order_size_model = order_size_model
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe = subscribe
        self.subscription_requested = False
        self.mid_list: List[float] = []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"

        self.assigned_exchange: Optional[int] = None
        if self.exchange_ids:
            self.assigned_exchange = self.random_state.choice(self.exchange_ids)

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if not can_trade or self.assigned_exchange is None:
            return

        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                self.assigned_exchange,
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=int(10e9),
                    depth=1,
                )
            )
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"

        elif not self.subscribe:
            self.get_current_spread(self.assigned_exchange, self.symbol)
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
            if sender_id == self.assigned_exchange:
                bid, _, ask, _ = self.get_known_bid_ask(self.assigned_exchange, self.symbol)
                self.analyse_and_place_order(bid, ask)
                self.set_wakeup(current_time + self.get_wake_frequency())
                self.state = "AWAITING_WAKEUP"

        # Subscription Mode
        elif (
                self.subscribe
                and self.state == "AWAITING_MARKET_DATA"
                and isinstance(message, MarketDataMsg)
        ):
            if sender_id == self.assigned_exchange:
                bids, asks = self.get_known_bid_ask(self.assigned_exchange, self.symbol, best=False)
                if bids and asks:
                    self.analyse_and_place_order(bids[0][0], asks[0][0])
            self.state = "AWAITING_MARKET_DATA"

    def analyse_and_place_order(self, bid: Optional[int], ask: Optional[int]) -> None:
        """
        Analyses market data from the assigned exchange to calculate momentum and places orders accordingly.
        This logic now perfectly mirrors the original single-exchange agent.
        """
        if bid and ask:
            self.mid_list.append((bid + ask) / 2)

            # Only proceed if we have enough data for a 50-period moving average.
            if len(self.mid_list) > 50:
                # Calculate the most recent 20- and 50-period MAs.
                avg_20 = MomentumAgentSE.ma(self.mid_list, n=20)[-1].round(2)
                avg_50 = MomentumAgentSE.ma(self.mid_list, n=50)[-1].round(2)

                if self.order_size_model is not None:
                    self.size = self.order_size_model.sample(
                        random_state=self.random_state
                    )

                if self.size > 0:
                    # Positive momentum: buy
                    if avg_20 >= avg_50:
                        self.place_limit_order(
                            exchange_id=self.assigned_exchange,
                            symbol=self.symbol,
                            quantity=self.size,
                            side=Side.BID,
                            limit_price=ask,
                        )
                    # Negative momentum: sell
                    else:
                        self.place_limit_order(
                            exchange_id=self.assigned_exchange,
                            symbol=self.symbol,
                            quantity=self.size,
                            side=Side.ASK,
                            limit_price=bid,
                        )

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

    # Re-instated static method to match original logic.
    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n