import logging
from itertools import permutations
from typing import Dict, Optional, List

import numpy as np

from abides_core import Message, NanosecondTime
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)


class ArbitrageAgent(TradingAgent):
    """
    An advanced arbitrage agent that dynamically sizes its trades based on
    market liquidity and actively manages its inventory to avoid accumulating
    excessive positions.
    """

    def __init__(
            self,
            id: int,
            symbol: str,
            starting_cash: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            wake_up_freq: NanosecondTime = 1_000_000_000,  # 1 second
            min_profit_margin: int = 1,  # Minimum profit in cents to execute a trade
            log_orders: bool = False,
            pov: float = 0.01,  # Percentage of available volume to use for order size
            max_inventory: int = 200,  # Maximum number of shares to hold before clearing
            exchange_ids: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            id,
            name=name,
            type=type,
            random_state=random_state,
            starting_cash=starting_cash,
            log_orders=log_orders,
            exchange_id=exchange_ids
        )
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival: bool = True
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.min_profit_margin = min_profit_margin

        self.pov = pov
        self.max_inventory = max_inventory
        # State management for collecting market data
        self.state = "INACTIVE"
        self.latest_spreads: Dict[int, QuerySpreadResponseMsg] = {}
        self.spreads_to_receive: int = 0


    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # The parent's kernel_starting method correctly finds all exchanges
        # and sets up the withdrawal fee map.
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        """On wakeup, the agent queries all exchanges to get a fresh view of the market."""
        can_trade = super().wakeup(current_time)
        if not can_trade:
            return

        # Set the next wakeup call.
        self.set_wakeup(current_time + self.get_wake_frequency())

        # Before seeking new trades, check if we need to clear out inventory
        total_holdings = self.get_holdings(self.symbol)
        if abs(total_holdings) > self.max_inventory:
            self.clear_excess_inventory(total_holdings)
            # TODO: Maybe after clearing inventory, wait for the next wakeup to reassess.
            # self.state = "INACTIVE"
            # return


        # Query all connected exchanges for their current spreads.
        if self.exchange_ids:
            self.spreads_to_receive = len(self.exchange_ids)
            self.latest_spreads = {}  # Clear old data
            for ex_id in self.exchange_ids:
                self.get_current_spread(ex_id, self.symbol)

            self.state = "AWAITING_SPREAD"

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        """Collects spread data and triggers the arbitrage logic when all data is present."""
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREAD" and isinstance(message, QuerySpreadResponseMsg):
            self.latest_spreads[sender_id] = message

            # Check if we have received responses from all exchanges.
            if len(self.latest_spreads) >= self.spreads_to_receive:
                self.state = "ANALYZING"
                self.find_and_execute_arbitrage()
                self.state = "INACTIVE"

    def find_and_execute_arbitrage(self) -> None:
        """The core logic to find and execute profitable arbitrage opportunities."""

        # Consider every possible pair of exchanges for an arbitrage trade.
        # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        for buy_exchange_id, sell_exchange_id in permutations(self.exchange_ids, 2):

            buy_market = self.latest_spreads.get(buy_exchange_id)
            sell_market = self.latest_spreads.get(sell_exchange_id)

            if not buy_market or not sell_market:
                continue

            # Get the best ask and bid prices from the exchanges selected.
            best_ask_price = buy_market.asks[0][0] if buy_market.asks else None
            best_bid_price = sell_market.bids[0][0] if sell_market.bids else None

            if best_ask_price is None or best_bid_price is None:
                continue

            # Arbitrage Condition
            if best_ask_price < best_bid_price:

                # Profit Calculation (Fee-Aware)
                gross_profit = best_bid_price - best_ask_price

                # Get the fee for withdrawing from the "buy" exchange to move it later.
                # A simple arbitrage agent assumes it must move the asset after buying.
                fee_structure = self.withdrawal_fees.get(buy_exchange_id, {})
                withdrawal_fee = fee_structure.get(self.symbol, fee_structure.get('default', 0))

                net_profit = gross_profit - withdrawal_fee
                print(
                    f"DEBUG ({self.name}): PROFITABLE SPREAD FOUND! Gross: {gross_profit}, Fee: {withdrawal_fee}, Net: {net_profit}")
                # 3. Check if the opportunity is profitable enough.
                if net_profit >= self.min_profit_margin:
                    #  Determine size based on available liquidity at the best price level.
                    buy_liquidity = buy_market.asks[0][1]
                    sell_liquidity = sell_market.bids[0][1]

                    # We can only trade the minimum of the two sides.
                    available_liquidity = min(buy_liquidity, sell_liquidity)

                    # Our order size is a percentage of this available liquidity.
                    dynamic_order_size = int(round(self.pov * available_liquidity))

                    if dynamic_order_size <= 0:
                        print(f"DEBUG ({self.name}): Opportunity found, but not enough liquidity to trade.")  # 🐛 DEBUG
                        continue # Not enough liquidity to trade
                    logger.info(
                        f"Arbitrage opportunity found! Buy on {buy_exchange_id} @ {best_ask_price}, "
                        f"Sell on {sell_exchange_id} @ {best_bid_price}. "
                        f"Net profit: {net_profit}. Size: {dynamic_order_size}"
                    )

                    print(
                        f"EXECUTION ({self.name}): Placing BUY order on Ex {buy_exchange_id} for {dynamic_order_size} @ {best_ask_price}")
                    print(
                        f"EXECUTION ({self.name}): Placing SELL order on Ex {sell_exchange_id} for {dynamic_order_size} @ {best_bid_price}")

                    # Execute the trades.
                    # Place the buy order on the cheap exchange.
                    self.place_limit_order(
                        buy_exchange_id, self.symbol, dynamic_order_size, Side.BID, best_ask_price
                    )
                    # Place the sell order on the expensive exchange.
                    self.place_limit_order(
                        sell_exchange_id, self.symbol, dynamic_order_size, Side.ASK, best_bid_price
                    )

                    # Once we execute a trade, we stop analyzing for this wakeup
                    return

    def clear_excess_inventory(self, total_holdings: int) -> None:
        """
        Places market orders to reduce inventory back to the max_inventory limit.
        """
        excess_inventory = abs(total_holdings) - self.max_inventory
        if excess_inventory <= 0:
            return
        print(f"DEBUG ({self.name}): Inventory limit exceeded. Clearing {excess_inventory} shares.")
        logger.info(f"{self.name} clearing excess inventory of {excess_inventory} shares.")

        # If we are long, we need to sell the excess.
        if total_holdings > 0:
            side = Side.ASK
            # Find the exchange with the best bid price to sell into.
            best_sell_price = float('-inf')
            best_sell_exchange = None
            for ex_id, spread in self.latest_spreads.items():
                if spread.bids and spread.bids[0][0] > best_sell_price:
                    best_sell_price = spread.bids[0][0]
                    best_sell_exchange = ex_id

            if best_sell_exchange is not None:
                print(
                    f"EXECUTION ({self.name}): Clearing inventory. Placing SELL market order on Ex {best_sell_exchange} for {excess_inventory} shares.")
                self.place_market_order(best_sell_exchange, self.symbol, excess_inventory, side)

        # If we are short, we need to buy back the excess.
        elif total_holdings < 0:
            side = Side.BID
            # Find the exchange with the best ask price to buy from.
            best_buy_price = float('inf')
            best_buy_exchange = None
            for ex_id, spread in self.latest_spreads.items():
                if spread.asks and spread.asks[0][0] < best_buy_price:
                    best_buy_price = spread.asks[0][0]
                    best_buy_exchange = ex_id

            if best_buy_exchange is not None:
                print(
                    f"EXECUTION ({self.name}): Clearing inventory. Placing BUY market order on Ex {best_buy_exchange} for {excess_inventory} shares.")
                self.place_market_order(best_buy_exchange, self.symbol, excess_inventory, side)

    def get_wake_frequency(self) -> NanosecondTime:
        """
        Returns the next wakeup interval.
        """
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            # Use an exponential distribution to model random arrivals
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

