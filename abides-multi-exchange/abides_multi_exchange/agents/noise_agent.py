import logging
from typing import Optional, List

import numpy as np

from abides_core import Message, NanosecondTime

from abides_markets.generators import OrderSizeGenerator
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class NoiseAgent(TradingAgent):
    """
    Noise agent implement simple strategy. The agent wakes up once and places 1 order.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "IBM",
        starting_cash: int = 100000,
        log_orders: bool = False,
        order_size_model: Optional[OrderSizeGenerator] = None,
        wakeup_time: Optional[NanosecondTime] = None,
        exchange_ids: Optional[List[int]] = None,
    ) -> None:

        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders, exchange_id=exchange_ids)

        self.wakeup_time: NanosecondTime = wakeup_time

        self.symbol: str = symbol  # symbol to trade

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time: Optional[NanosecondTime] = None

        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )

        self.order_size_model = order_size_model  # Probabilistic model for order size

        # Agent
        self.target_exchange: Optional[int] = None

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Just use parent method.
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)
        # print(f"DEBUG ({self.name} @ {current_time}): Woke up. Current state: {self.state}")

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                if self.exchange_ids:
                    self.target_exchange = self.random_state.choice(self.exchange_ids)
                    # print(f"DEBUG ({self.name}): I have chosen to trade on Exchange {self.target_exchange}.")
                    logger.debug(f"{self.name} has chosen to trade on Exchange {self.target_exchange}")
                else:
                    logger.warning(f"{self.name} has no exchange to trade on.")
                    self.state = "INACTIVE"
                    return

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        if self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            return

        if self.mkt_closed and self.symbol not in self.daily_close_price:
            self.get_current_spread(self.target_exchange, self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        if type(self) == NoiseAgent:
            # print(
            #     f"DEBUG ({self.name} @ {current_time}): Requesting spread from my target Exchange {self.target_exchange}.")
            self.get_current_spread(self.target_exchange, self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    def placeOrder(self) -> None:
        # place order in random direction at a mid
        # print(f"{self.name}: Preparing to order.")
        buy_indicator = self.random_state.randint(0, 1 + 1)

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.target_exchange, self.symbol)

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)
        # print(f"{self.name}: got order size model and size is {self.size}, buy_indicator {buy_indicator}, ask {ask} bid {bid}.")

        if self.size > 0:
            if buy_indicator == 1 and ask:
                # print(
                #     f"DEBUG ({self.name}): Placing BUY limit order for {self.size} shares @ {ask} on Exchange {self.target_exchange}")
                self.place_limit_order(self.target_exchange, self.symbol, self.size, Side.BID, ask)
            elif not buy_indicator and bid:
                # print(
                #     f"DEBUG ({self.name}): Placing SELL limit order for {self.size} shares @ {ask} on Exchange {self.target_exchange}")
                self.place_limit_order(self.target_exchange, self.symbol, self.size, Side.ASK, bid)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)
        # print(f"DEBUG ({self.name} @ {current_time}): Received spread response from Exchange {sender_id}. Preparing to place order.")

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.
            # print(f"DEBUG ({self.name} @ {current_time}): Received response from Exchange {sender_id}. mkt is open. is it {self.mkt_closed}")

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed.get(sender_id, True):
                    return
                # print(
                #     f"DEBUG ({self.name} @ {current_time}): Received spread response from Exchange {sender_id}. mkt is open.")

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = "AWAITING_WAKEUP"

    # Internal state and logic specific to this agent subclass.

    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=0, high=100)
