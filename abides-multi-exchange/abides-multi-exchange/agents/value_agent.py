import logging
from typing import Optional, Dict

import numpy as np

from abides_core import Message, NanosecondTime

from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class ValueAgent(TradingAgent):
    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "IBM",
        starting_cash: int = 100_000,
        sigma_n: float = 10_000,
        r_bar: int = 100_000,
        kappa: float = 0.05,
        sigma_s: float = 100_000,
        order_size_model=None,
        lambda_a: float = 0.005,
        log_orders: float = False,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        # Store important parameters particular to the ZI agent.
        self.symbol: str = symbol  # symbol to trade
        self.sigma_n: float = sigma_n  # observation noise variance
        self.r_bar: int = r_bar  # true mean fundamental value
        self.kappa: float = kappa  # mean reversion parameter
        self.sigma_s: float = sigma_s  # shock variance
        self.lambda_a: float = lambda_a  # mean arrival rate of ZI agents

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent maintains two priors: r_t and sigma_t (value and error estimates).
        self.r_t: int = r_bar
        self.sigma_t: float = 0

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time: Optional[NanosecondTime] = None

        # Percent of time that the agent will aggress the spread
        self.percent_aggr: float = 0.1

        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size

        self.depth_spread: int = 2

        # A dictionary to store the most recent spread data from each exchange
        self.latest_spreads: Dict[int, QuerySpreadResponseMsg] = {}

        # A counter to know when we have received all expected spread responses
        self.spreads_to_receive: int = 0

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Just let the Parent class handle the logic here.
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.set_wakeup(current_time + int(round(delta_time)))

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        self.cancel_all_orders()

        if type(self) == ValueAgent:
            # Must account for multiple exchanges
            if self.exchange_ids:
                self.spreads_to_receive = len(self.exchange_ids)
                self.latest_spreads = {}
                for ex_id in self.exchange_ids:
                    self.get_current_spread(ex_id, self.symbol)
                self.state = "AWAITING_SPREAD"
            else:
                self.state = "INACTIVE"
        else:
            self.state = "ACTIVE"

    def updateEstimates(self) -> int:
        # Called by a background agent that wishes to obtain a new fundamental observation,
        # update its internal estimation parameters, and compute a new total valuation for the
        # action it is considering.

        # The agent obtains a new noisy observation of the current fundamental value
        # and uses this to update its internal estimates in a Bayesian manner.

        obs_t = self.oracle.observe_price(
            self.symbol,
            self.current_time,
            sigma_n=self.sigma_n,
            random_state=self.random_state,
        )

        logger.debug("{} observed {} at {}", self.name, obs_t, self.current_time)

        # Update internal estimates of the current fundamental value and our error of same.

        # If this is our first estimate, treat the previous wake time as "market open".
        if self.prev_wake_time is None:
            self.prev_wake_time = self.mkt_open

        # First, obtain an intermediate estimate of the fundamental value by advancing
        # time from the previous wake time to the current time, performing mean
        # reversion at each time step.

        # delta must be integer time steps since last wake
        delta = self.current_time - self.prev_wake_time

        # Update r estimate for time advancement.
        r_tprime = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_tprime += ((1 - self.kappa) ** delta) * self.r_t

        # Update sigma estimate for time advancement.
        sigma_tprime = ((1 - self.kappa) ** (2 * delta)) * self.sigma_t
        sigma_tprime += (
            (1 - (1 - self.kappa) ** (2 * delta)) / (1 - (1 - self.kappa) ** 2)
        ) * self.sigma_s

        # Apply the new observation, with "confidence" in the observation inversely proportional
        # to the observation noise, and "confidence" in the previous estimate inversely proportional
        # to the shock variance.
        self.r_t = (self.sigma_n / (self.sigma_n + sigma_tprime)) * r_tprime
        self.r_t += (sigma_tprime / (self.sigma_n + sigma_tprime)) * obs_t

        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t)

        # Now having a best estimate of the fundamental at time t, we can make our best estimate
        # of the final fundamental (for time T) as of current time t.  Delta is now the number
        # of time steps remaining until the simulated exchange closes.
        delta = max(0, (self.mkt_close - self.current_time))

        # IDEA: instead of letting agent "imagine time forward" to the end of the day,
        #       impose a maximum forward delta, like ten minutes or so.  This could make
        #       them think more like traders and less like long-term investors.  Add
        #       this line of code (keeping the max() line above) to try it.
        # delta = min(delta, 1000000000 * 60 * 10)

        r_T = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_T += ((1 - self.kappa) ** delta) * self.r_t

        # Our final fundamental estimate should be quantized to whole units of value.
        r_T = int(round(r_T))

        # Finally (for the final fundamental estimation section) remember the current
        # time as the previous wake time.
        self.prev_wake_time = self.current_time

        logger.debug(
            "{} estimates r_T = {} as of {}", self.name, r_T, self.current_time
        )

        return r_T

    def analyze_and_place_order(self) -> None:
        """
        Analyzes spreads from all exchanges to find the best market,
        then applies a smart passive pricing strategy to place an order.
        """
        # 1. Get the agent's estimate of the fundamental value.
        r_T = self.updateEstimates()

        # 2. Find the single best bid and ask price across all exchanges.
        best_global_bid = float('-inf')
        best_global_ask = float('inf')
        best_bid_exchange = None
        best_ask_exchange = None

        for ex_id, spread_msg in self.latest_spreads.items():
            if spread_msg.asks:
                best_ask_on_exchange = spread_msg.asks[0][0]
                if best_ask_on_exchange < best_global_ask:
                    best_global_ask = best_ask_on_exchange
                    best_ask_exchange = ex_id

            if spread_msg.bids:
                best_bid_on_exchange = spread_msg.bids[0][0]
                if best_bid_on_exchange > best_global_bid:
                    best_global_bid = best_bid_on_exchange
                    best_bid_exchange = ex_id

        # If we couldn't find a valid spread on any exchange, do nothing.
        if best_bid_exchange is None or best_ask_exchange is None:
            logger.debug(f"{self.name} could not find a valid spread on any exchange.")
            return

        # 3. Use the best global prices to make a decision (same as original logic).
        midpoint = (best_global_ask + best_global_bid) / 2

        # Determine the side and the baseline price for the order.
        if r_T < midpoint:
            # Fundamental belief is that price will go down, so we want to sell.
            side = Side.ASK
            target_exchange = best_bid_exchange
            baseline_price = best_global_bid
        elif r_T >= midpoint:
            # Fundamental belief is that price will go up, so we want to buy.
            side = Side.BID
            target_exchange = best_ask_exchange
            baseline_price = best_global_ask
        else:
            # No clear opportunity.
            return

        # 4. Apply the "smart passive" pricing logic (same as original logic).
        spread = abs(best_global_ask - best_global_bid)

        # Decide whether to be aggressive (cross the spread) or passive.
        if self.random_state.rand() < self.percent_aggr:
            # Aggressive: place the order at the best available price.
            limit_price = baseline_price
        else:
            # Passive: adjust the price deeper into the book to maximize surplus.
            adjust_int = self.random_state.randint(0,
                                                   min(9223372036854775807 - 1, self.depth_spread * spread)
                                                   )

            if side == Side.BID:
                limit_price = baseline_price - adjust_int
            else:  # Side is ASK
                limit_price = baseline_price + adjust_int

        # 5. Place the final, intelligently priced order.
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0:
            logger.debug(
                f"{self.name} placing {side.value} order for {self.size} on exchange {target_exchange} at price {limit_price}")
            self.place_limit_order(target_exchange, self.symbol, self.size, side, limit_price)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.
                self.latest_spreads[sender_id] = message
                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed[sender_id]:
                    return
                if len(self.latest_spreads) >= self.spreads_to_receive:
                    # We now have the information needed to place a limit order with the eta
                    # strategic threshold parameter.
                    self.analyze_and_place_order()
                    self.state = "AWAITING_WAKEUP"

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def get_wake_frequency(self) -> NanosecondTime:
        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        return int(round(delta_time))
