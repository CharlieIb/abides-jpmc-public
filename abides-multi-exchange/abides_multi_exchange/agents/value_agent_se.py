import logging
from typing import Optional, Dict, List

import numpy as np

from abides_core import Message, NanosecondTime

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)


class ValueAgentSE(TradingAgent):
    def __init__(
            self,
            id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            symbol: str = "ABM",
            starting_cash: int = 100_000,
            sigma_n: float = 10_000,
            r_bar: int = 100_000,
            kappa: float = 0.05,
            sigma_s: float = 100_000_000,
            order_size_model=None,
            lambda_a: float = 0.005,
            log_orders: bool = False,
            exchange_ids: Optional[List[int]] = None,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders, exchange_id=exchange_ids)

        # Store important parameters from the multi-exchange version.
        self.symbol: str = symbol
        self.sigma_n: float = sigma_n
        self.r_bar: int = r_bar
        self.kappa: float = kappa
        self.sigma_s: float = sigma_s
        self.lambda_a: float = lambda_a

        #  Logic from the base single-exchange agent
        self.trading: bool = False
        self.state: str = "AWAITING_WAKEUP"
        self.r_t: int = r_bar
        self.sigma_t: float = 0
        self.prev_wake_time: Optional[NanosecondTime] = None
        self.percent_aggr: float = 0.1
        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model
        self.depth_spread: int = 2

        # Assign agent to a single exchange for its lifecycle
        self.assigned_exchange: Optional[int] = None
        if self.exchange_ids:
            self.assigned_exchange = self.random_state.choice(self.exchange_ids)

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Using the base agent's kernel_stopping for final valuation logging.
        super().kernel_stopping()
        # The parent class now handles the surplus calculation logic,
        # but if you need the original detailed logging, it can be re-added here.

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times.
        super().wakeup(current_time)
        self.state = "INACTIVE"

        if self.assigned_exchange is None:
            return

        mkt_open = self.mkt_open.get(self.assigned_exchange)
        mkt_close = self.mkt_close.get(self.assigned_exchange)

        if not mkt_open or not mkt_close:
            return
        else:
            if not self.trading:
                self.trading = True
                logger.debug(f"{self.name} is ready to start trading.")

        mkt_closed_for_exchange = self.mkt_closed.get(self.assigned_exchange, False)

        if mkt_closed_for_exchange and (self.symbol in self.daily_close_price.get(self.symbol, {})):
            return

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.set_wakeup(current_time + int(round(delta_time)))

        if mkt_closed_for_exchange and (self.symbol not in self.daily_close_price.get(self.symbol, {})):
            self.get_current_spread(self.assigned_exchange, self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        self.cancel_all_orders()
        if type(self) == ValueAgentSE:
            self.get_current_spread(self.assigned_exchange, self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    def updateEstimates(self) -> int:
        obs_t = self.oracle.observe_price(
            self.symbol,
            self.current_time,
            sigma_n=self.sigma_n,
            random_state=self.random_state,
        )
        logger.debug(f"{self.name} observed {obs_t} at {self.current_time}")

        mkt_open = self.mkt_open.get(self.assigned_exchange)
        mkt_close = self.mkt_close.get(self.assigned_exchange)

        if self.prev_wake_time is None:
            self.prev_wake_time = mkt_open

        delta = self.current_time - self.prev_wake_time

        r_tprime = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_tprime += ((1 - self.kappa) ** delta) * self.r_t

        sigma_tprime = (((1 - self.kappa) ** (2 * delta)) * self.sigma_t )
        sigma_tprime += ((1 - (1 - self.kappa) ** (2 * delta)) / (1 - (1 - self.kappa) ** 2)) * self.sigma_s

        self.r_t = (self.sigma_n / (self.sigma_n + sigma_tprime)) * r_tprime
        self.r_t += (sigma_tprime / (self.sigma_n + sigma_tprime)) * obs_t

        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t)

        delta_to_close = max(0, (mkt_close - self.current_time))
        r_T = (1 - (1 - self.kappa) ** delta_to_close) * self.r_bar
        r_T += (((1 - self.kappa) ** delta_to_close) * self.r_t)
        r_T = int(round(r_T))

        self.prev_wake_time = self.current_time
        logger.debug(f"{self.name} estimates r_T = {r_T} as of {self.current_time}")
        return r_T

    def place_order(self) -> None:
        # This method's logic is taken directly from the base single-exchange agent.
        r_T = self.updateEstimates()

        bid, _, ask, _ = self.get_known_bid_ask(self.assigned_exchange, self.symbol)

        if bid and ask:
            mid = int((ask + bid) / 2)
            spread = abs(ask - bid)

            if self.random_state.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = self.random_state.randint(0, min(spread, self.depth_spread * spread))

            if r_T < mid:
                side = Side.ASK
                price = bid + adjust_int
            elif r_T >= mid:
                side = Side.BID
                price = ask - adjust_int
        else:
            # If no spread, default to placing order around fundamental estimate.
            side = Side.BID if self.random_state.randint(2) else Side.ASK
            price = r_T

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0:
            self.place_limit_order(self.assigned_exchange, self.symbol, self.size, side, price)

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREAD":
            if isinstance(message, QuerySpreadResponseMsg) and sender_id == self.assigned_exchange:

                mkt_closed_for_exchange = self.mkt_closed.get(self.assigned_exchange, False)
                if mkt_closed_for_exchange:
                    return

                self.place_order()
                self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        return int(round(delta_time))