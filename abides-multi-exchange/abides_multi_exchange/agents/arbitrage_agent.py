import logging
from itertools import permutations
from typing import Dict, Optional, List

import numpy as np

from abides_core import Message, NanosecondTime
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent
from ..messages import CompleteTransferMsg

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
            min_profit_margin: float = 1,  # Minimum profit in cents to execute a trade
            log_orders: bool = False,
            pov: float = 0.1,  # Percentage of available volume to use for order size
            max_inventory: int = 100000,  # Maximum number of shares to hold before clearing
            exchange_ids: Optional[List[int]] = None,
            trading_fee_percentage: float = 0.001, # 0.1% fee per trade
            rebalance_threshold: int = 9000, # Difference between exchange holdings and average
            initial_symbol_holdings_on_each_exchange: int = 10000
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
        self.trading_fee_percentage = trading_fee_percentage

        self.pov = pov
        self.max_inventory = max_inventory
        self.rebalance_threshold = rebalance_threshold

        self.rebalance_check_interval: NanosecondTime = 3600_000_000_000  # Check every hour.
        self.last_rebalance_check_time: NanosecondTime = 0

        # State management for collecting market data
        self.state = "INACTIVE"
        self.latest_spreads: Dict[int, QuerySpreadResponseMsg] = {}
        self.spreads_to_receive: int = 0
        self.initial_holdings = initial_symbol_holdings_on_each_exchange

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # The parent's kernel_starting method correctly finds all exchanges
        # and sets up the withdrawal fee map.
        super().kernel_starting(start_time)
        for ex_id in self.exchange_ids:
            self.holdings_by_exchange[ex_id][self.symbol] = self.initial_holdings

    def wakeup(self, current_time: NanosecondTime) -> None:
        """On wakeup, the agent queries all exchanges to get a fresh view of the market."""
        can_trade = super().wakeup(current_time)
        if not can_trade:
            return

        # Set the next wakeup call.
        self.set_wakeup(current_time + self.get_wake_frequency())

        # Check if it is time to rebalance
        if self.time_to_rebalance(current_time):
            if self.check_inventory_skew():
                self.rebalance_funds()
                return


        # Query all connected exchanges for their current spreads.
        if self.exchange_ids:
            self.spreads_to_receive = len(self.exchange_ids)
            self.latest_spreads = {}  # Clear old data
            for ex_id in self.exchange_ids:
                self.get_current_spread(ex_id, self.symbol)

            self.state = "AWAITING_SPREADS"

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        """Collects spread data and triggers the arbitrage logic when all data is present."""
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREADS" and isinstance(message, QuerySpreadResponseMsg):
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

            if not buy_market or not sell_market or not buy_market.asks or not sell_market.bids:
                continue

            # Get the best ask and bid prices from the exchanges selected.
            best_ask_price = buy_market.asks[0][0]
            best_bid_price = sell_market.bids[0][0]

            # As the agent rebalances periodically, it models the trading fee percentage
            # rather than use the actual one --- this can be adjusted for realism
            effective_buy_price = best_ask_price * (1 + self.trading_fee_percentage)
            effective_sell_price = best_bid_price * (1 - self.trading_fee_percentage)

            net_profit_per_share = effective_sell_price - effective_buy_price

            if net_profit_per_share >= self.min_profit_margin:
                buy_liquidity = buy_market.asks[0][1]
                sell_liquidity = sell_market.bids[0][1]
                available_liquidity = min(buy_liquidity, sell_liquidity)
                dynamic_order_size = int(round(self.pov * available_liquidity))

                if dynamic_order_size <= 0:
                    continue # Not enough liquidity to trade

                cash = self.cash
                holdings_on_sell_exchange = self.get_holdings_by_exchange(
                    self.symbol, sell_exchange_id)

                # This is a simplified check. A robust agent would track cash/holdings per exchange.
                if cash < dynamic_order_size * best_ask_price:
                    logger.info(f"Not enough cash to execute buy.")
                    continue

                if holdings_on_sell_exchange < dynamic_order_size:
                    logger.info(f"Not enough holdings of {self.symbol} on Ex {sell_exchange_id} to execute sell.")
                    continue

                logger.info(
                    f"Arbitrage opportunity found! Buy on {buy_exchange_id} @ {best_ask_price}, "
                    f"Sell on {sell_exchange_id} @ {best_bid_price}. "
                    f"Net profit: {net_profit_per_share}. Size: {dynamic_order_size}"
                )

                # print(
                #     f"EXECUTION ({self.name}): Placing BUY order on Ex {buy_exchange_id} for {dynamic_order_size} @ {best_ask_price}")
                # print(
                #     f"EXECUTION ({self.name}): Placing SELL order on Ex {sell_exchange_id} for {dynamic_order_size} @ {best_bid_price}")

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

    def rebalance_funds(self) -> None:
        """
        Triggered periodically (e.g. daily) or when inventories become too skewed.
        It involves making on-chain transfers
        """
        logger.info(f"Agent {self.name} is assessing the need for rebalancing.")

        # Exit if rebalancing is not possible or necessary
        if not self.exchange_ids or len(self.exchange_ids) < 2:
            return

        # Find total holdings and holdings by exchange
        total_holdings = self.get_holdings(self.symbol)
        if total_holdings <= 0:
            logger.info(f"Agent {self.name} has no holdings.")
            return

        current_holdings = {}
        for ex_id in self.exchange_ids:
            current_holdings[ex_id] = self.get_holdings_by_exchange(self.symbol, ex_id)

        target_per_exchange = total_holdings / len(self.exchange_ids)
        logger.info(
            f"Total Holdings: {total_holdings} {self.symbol}"
            f"Target per exchange: {target_per_exchange:.2f}"
        )

        surpluses = {}
        deficits = {}

        rebalance_min_amount = 7500 # This must be very high, as the withdrawal fee is approx the cost of 1 share

        for ex_id, holding in current_holdings.items():
            diff = holding - target_per_exchange
            if diff > rebalance_min_amount:
                surpluses[ex_id] = diff
            elif diff < -rebalance_min_amount:
                deficits[ex_id] = -diff # Stored as positive number rep. need

        if not deficits or not surpluses:
            logger.info("Holdings are already balanced. No transfers needed")
            return

        # plan transfers, prioritising withdrawals from cheapest exchange
        surplus_fees = []
        for ex_id in surpluses:
            fee_structure = self.withdrawal_fees.get(ex_id, {})
            fee = fee_structure.get(self.symbol, fee_structure.get('default', float('inf')))
            surplus_fees.append({'id': ex_id, 'fee': fee})

        sorted_surplus_exchanges = sorted(surplus_fees, key=lambda k: k['fee'])
        transfers_to_make = []

        for deficit_ex_id, needed_amount in deficits.items():
            for surplus_ex in sorted_surplus_exchanges:
                if needed_amount <= 0:
                    break
                surplus_ex_id, available_surplus = surplus_ex['id'], surpluses[surplus_ex['id']]
                if available_surplus > 0:
                    transfer_amount = min(needed_amount, available_surplus)
                    transfers_to_make.append({"from": surplus_ex_id, "to": deficit_ex_id, "amount": transfer_amount,
                                              "fee": surplus_ex['fee']})
                    needed_amount -= transfer_amount
                    surpluses[surplus_ex_id] -= transfer_amount

        # Transfer execution with delay
        logger.info("Initiating delayed rebalancing transfers..")
        for transfer in transfers_to_make:
            from_ex, to_ex, amount, fee = transfer["from"], transfer["to"], transfer["amount"], transfer["fee"]

            self.cash -= fee
            try:
                self.holdings_by_exchange[from_ex][self.symbol] -= amount
            except (AttributeError, KeyError):
                logger.error(f"Cannot withdraw from Ex {from_ex} Agent state `holdings_by_exchange` is missing or invalid.")
                continue

            transfer_delay = self._get_random_transfer_delay()
            completion_msg = CompleteTransferMsg(
                to_exchange=to_ex,
                symbol=self.symbol,
                size=amount,
            )

            print(
                f"EXECUTION ({self.name}): Initiating transfer of {amount:.2f} {self.symbol} "
                f"from Ex {from_ex} to Ex {to_ex}. Fee: {fee}. "
                f"Arrival in approx {transfer_delay / (60 * 1e9):.2f} min(s)."
            )

            self.send_message(
                recipient_id=self.id,
                message=completion_msg,
                delay=transfer_delay
            )

    def time_to_rebalance(self, current_time: NanosecondTime) -> bool:
        """
        Determines if enough time has passed to check for rebalancing.
        """
        if self.last_rebalance_check_time == 0:
            # First time checking, so proceed.
            self.last_rebalance_check_time = current_time
            return True

        if current_time - self.last_rebalance_check_time > self.rebalance_check_interval:
            self.last_rebalance_check_time = current_time
            return True

        return False

    def check_inventory_skew(self) -> bool:
        """
        Checks if the inventory of the symbol is too skewed across exchanges.
        Returns True if the skew exceeds the rebalance_threshold, False otherwise.
        """
        total_holdings = self.get_holdings(self.symbol)
        if total_holdings <= 0 or not self.exchange_ids:
            return False

        # Get target even distribution.
        current_holdings = {ex_id: self.get_holdings_by_exchange(self.symbol, ex_id) for ex_id in self.exchange_ids}
        target_per_exchange = total_holdings / len(self.exchange_ids)

        # Check if any exchange deviates from the target by more than the threshold.
        # If so rebalance
        for holding in current_holdings.values():
            if abs(holding - target_per_exchange) > self.rebalance_threshold:
                logger.info(
                    f"Inventory skew detected. Target: {target_per_exchange:.2f}, "
                    f"Actual: {holding}. Threshold: {self.rebalance_threshold}. Triggering rebalance."
                )
                # print(f"DEBUG ({self.name}): Inventory skew detected. Triggering rebalance.")
                return True

        return False

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

