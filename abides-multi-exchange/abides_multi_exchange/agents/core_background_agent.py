from collections import deque
from copy import deepcopy
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.generators import ConstantTimeGenerator, InterArrivalTimeGenerator
from abides_core.utils import str_to_ns
from .trading_agent import TradingAgent
from abides_markets.messages.marketdata import (
    MarketDataMsg,
    L2DataMsg,
    L2SubReqMsg,
    TransactedVolDataMsg,
    TransactedVolSubReqMsg,
)
from ..messages import (
    TradeDataSubReqMsg,
    TradeDataMsg
)
from abides_markets.orders import Order, Side
from abides_multi_exchange.messages import CompleteTransferMsg


class CoreBackgroundAgent(TradingAgent):
    """
    A base class for agents that interface with an external environment (e.g., an RL Gym).
    This agent is adapted for a multi-exchange simulation with segregated asset holdings.
    It gathers data from all connected exchanges and requires actions to specify a target exchange.
    """
    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        subscribe_freq: int = int(1e8),
        lookback_period: Optional[int] = None, # for volume subscription
        subscribe: bool = True,
        subscribe_num_levels: Optional[int] = None,
        wakeup_interval_generator: InterArrivalTimeGenerator = ConstantTimeGenerator(
            step_duration=str_to_ns("1min")
        ),
        order_size_generator=None, # TODO: not sure about this one
        state_buffer_length: int = 2,
        market_data_buffer_length: int = 5,
        first_interval: Optional[NanosecondTime] = None,
        log_orders: bool = False,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        # Note: The parent __init__ handles starting_cash correctly for the new self.cash attribute.
        super().__init__(
            id,
            starting_cash=starting_cash,
            log_orders=log_orders,
            name=name,
            type=type,
            random_state=random_state
        )
        self.symbol: str = symbol
        self.subscribe_freq: int = subscribe_freq
        self.subscribe: bool = subscribe
        self.subscribe_num_levels: int = subscribe_num_levels
        self.first_interval: Optional[NanosecondTime] = first_interval
        self.wakeup_interval_generator: InterArrivalTimeGenerator = wakeup_interval_generator
        self.order_size_generator = order_size_generator # TODO: no idea here for typing

        if hasattr(self.wakeup_interval_generator, "random_generator"):
            self.wakeup_interval_generator.random_generator = self.random_state
        if self.order_size_generator:
            self.order_size_generator.random_generator = self.random_state

        self.state_buffer_length: int = state_buffer_length
        self.market_data_buffer_length: int = market_data_buffer_length

        self.lookback_period: NanosecondTime = self.wakeup_interval_generator.mean()

        # internal variables
        self.has_subscribed: bool = False
        self.episode_executed_orders: List[Order] = []
        self.inter_wakeup_executed_orders: List[Order] = []
        self.parsed_episode_executed_orders: List[Tuple[int, int]] = []
        self.parsed_inter_wakeup_executed_orders: List[Tuple[int, int]] = []
        self.parsed_mkt_data_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.market_data_buffer_length)
        self.parsed_volume_data_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.market_data_buffer_length)
        self.parsed_trade_data_buffer: Deque[List[Dict[str, Any]]] = deque(maxlen=self.market_data_buffer_length)
        self.raw_state: Deque[Dict[str, Any]] = deque(maxlen=self.state_buffer_length)
        self.order_status: Dict[int, Dict[str, Any]] = {}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # Parent class handles finding all exchanges and setting up withdrawal fees.
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> bool:
        """On wakeup, the agent subscribes to all exchanges (if it hasn't already)
           and then calls its main logic loop."""
        #print(f"DEBUG: CoreBackgroundAgent ({self.id}) waking up at time {current_time}")
        ready_to_trade = super().wakeup(current_time)

        if not self.has_subscribed and self.exchange_ids:
            #print(f"DEBUG: CoreBackgroundAgent ({self.id}) is attempting to subscribe now.")
            # Subscribe to all available exchanges
            for ex_id in self.exchange_ids:
                if self.subscribe:
                    super().request_data_subscription(
                        ex_id,
                        L2SubReqMsg(
                            symbol=self.symbol,
                            freq=self.subscribe_freq,
                            depth=self.subscribe_num_levels,
                        )
                    )
                    super().request_data_subscription(
                        ex_id,
                        TransactedVolSubReqMsg(
                            symbol=self.symbol,
                            lookback=self.lookback_period,
                        )
                    )
                    super().request_data_subscription(
                        ex_id,
                        TradeDataSubReqMsg(
                            symbol=self.symbol,
                            freq=self.subscribe_freq
                        )
                    )
            self.has_subscribed = True

        # Check if any market is open
        # The agent can act as long as at least one market is open.
        is_any_market_open = any(self.mkt_open.get(ex_id) and not self.mkt_closed.get(ex_id, True) for ex_id in self.exchange_ids)

        if is_any_market_open:
            # Schedule the next wakeup call for this agent
            self.set_wakeup(current_time + self.get_wake_frequency())
            raw_state = self.act_on_wakeup()
            # TODO: wakeupfunction should return bool
            return raw_state # Return state to the Gym Env

    def act_on_wakeup(self):
        # This method should be implemented by the specific RL agent.
        raise NotImplementedError

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        """Processes message from an exchange and stores data tagged by exchange ID."""
        # TODO: will probe need to see for transacted volume if we enrich the state
        super().receive_message(current_time, sender_id, message)
        if self.subscribe and isinstance(message, MarketDataMsg):
            if isinstance(message, L2DataMsg):
                parsed_mkt_data = self.get_parsed_mkt_data(sender_id, message)
                self.parsed_mkt_data_buffer.append(parsed_mkt_data)
            elif isinstance(message, TransactedVolDataMsg):
                parsed_volume_data = self.get_parsed_volume_data(sender_id, message)
                self.parsed_volume_data_buffer.append(parsed_volume_data)
            elif isinstance(message, TradeDataMsg):
                # --- NEW --- : pipeline to handle trade data
                if message.trades:
                    self.parsed_trade_data_buffer.append(message.trades)



    def get_wake_frequency(self) -> NanosecondTime:
        """
        Returns the next wakeup interval from the generator.
        Handles the special case for the first interval.
        """
        # First wakeup interval from open.
        if self.first_interval is not None:
            time_to_wakeup = self.first_interval
            self.first_interval = None  # Ensure it's only used once
            return time_to_wakeup
        else:
            return self.wakeup_interval_generator.next()

    def apply_actions(self, actions: List[Dict[str, Any]]) -> None:
        """
        Takes actions from the controlling environment.
        The `actions` dictionary must now include an `exchange_id`.
        """
        # TODO: Add cancel in actions
        for action in actions:
            action_type = action.get("type")
            if action_type in ["MKT", "LMT"]:
                exchange_id = action.get("exchange_id")
                if exchange_id is None:
                    raise ValueError(f"Action {action} must include an 'exchange_id'")
                side = Side.BID if action["direction"] == "BUY" else Side.ASK

                if action["type"] == "MKT":
                    self.place_market_order(exchange_id, self.symbol, action["size"], side)
                elif action["type"] == "LMT":
                    self.place_limit_order(exchange_id, self.symbol, action["size"], side, action["limit_price"])
            elif action_type == "TFR":
                from_ex = action.get("from_exchange")
                to_ex = action.get("to_exchange")
                size = action.get("size")

                # Check if the agent has enough holdings to transfer
                current_holdings = self.get_holdings_by_exchange(self.symbol, exchange_id=from_ex)
                if current_holdings < size:
                    print(f"{self.name} failed to transfer {size} shares from Ex {from_ex}. "
                                   f"Holdings: {current_holdings}")
                    continue

                # Apply fee structure
                fee_structure = self.withdrawal_fees.get(from_ex, {})
                withdrawal_fee = fee_structure.get(self.symbol, fee_structure.get('default', 0))
                self.cash -= withdrawal_fee
                self.holdings_by_exchange[from_ex][self.symbol] -= size

                # Creates a random transfer delay between specified amounts
                transfer_delay = self._get_random_transfer_delay()

                print(f"DEBUG ({self.name}): Initiating transfer of {size} shares from Ex {from_ex} to {to_ex}. "
                      f"Fee: ${withdrawal_fee/100:.2f}, Delay: {transfer_delay/(60*1_000_000_000):.2f} min(s)")

                # 3. Schedule the completion of the transfer after a delay
                completion_msg = CompleteTransferMsg(
                    to_exchange=to_ex,
                    symbol=self.symbol,
                    size=size
                )

                self.send_message(
                    recipient_id=self.id,
                    message=completion_msg,
                    delay=transfer_delay
                )

            elif action_type == "CCL_ALL":
                self.cancel_all_orders() # This parent method correctly cancels across all exchanges.
            else:
                raise ValueError(f"Action Type '{action_type}' is not supported")

    def update_raw_state(self) -> None:
        parsed_mkt_data_buffer = deepcopy(self.parsed_mkt_data_buffer)
        internal_data = self.get_internal_data()
        parsed_volume_data_buffer = deepcopy(self.parsed_volume_data_buffer)
        parsed_trade_data_buffer = deepcopy(self.parsed_trade_data_buffer)
        new = {
            "parsed_mkt_data": parsed_mkt_data_buffer,
            "internal_data": internal_data,
            "parsed_volume_data": parsed_volume_data_buffer,
            "parsed_trade_data": parsed_trade_data_buffer,
        }
        self.raw_state.append(new)

    def get_raw_state(self) -> Deque:
        return self.raw_state

    def get_parsed_mkt_data(self, exchange_id: int, message: L2DataMsg) -> Dict[str, Any]:
        """Parses L2 market data and tags it with the exchange ID."""
        return {
            "exchange_id": exchange_id, # Tag data with its source
            "bids": message.bids,
            "asks": message.asks,
            "last_transaction": message.last_transaction,
            "exchange_ts": message.exchange_ts,
        }

    def get_parsed_volume_data(self, exchange_id: int, message: TransactedVolDataMsg) -> Dict[str, Any]:
        """Parses volume data and tags it with the exchange ID."""
        return {
            "exchange_id": exchange_id, # Tag data with its source
            "last_transaction": message.last_transaction,
            "exchange_ts": message.exchange_ts,
            "bid_volume": message.bid_volume,
            "ask_volume": message.ask_volume,
            "total_volume": message.bid_volume + message.ask_volume,
        }

    def get_internal_data(self) -> Dict[str, Any]:
        """Gathers the agent's internal state, adapted for the new data structures."""
        # --- MODIFICATION: Use new segregated holdings and cash attributes ---
        return {
            "holdings_by_exchange": deepcopy(self.holdings_by_exchange),
            "total_holdings": self.get_holdings(self.symbol), # Parent method sums across exchanges
            "cash": self.cash, # Use the new self.cash attribute
            "starting_cash": self.starting_cash,
            "inter_wakeup_executed_orders": self.inter_wakeup_executed_orders,
            "episode_executed_orders": self.episode_executed_orders,
            "parsed_episode_executed_orders": self.parsed_episode_executed_orders,
            "parsed_inter_wakeup_executed_orders": self.parsed_inter_wakeup_executed_orders,
            "current_time": self.current_time,
            "order_status": self.order_status,
            "mkt_open_times": self.mkt_open, # Now a dict of times
            "mkt_close_times": self.mkt_close, # Now a dict of times
            "withdrawal_fees": self.withdrawal_fees,
        }

    def order_executed(self, exchange_id: int, order: Order) -> None:
        super().order_executed(exchange_id, order)
        # parsing of the order message
        executed_qty = order.quantity
        executed_price = order.fill_price
        assert executed_price is not None
        order_id = order.order_id
        # step lists
        self.inter_wakeup_executed_orders.append(order)
        self.parsed_inter_wakeup_executed_orders.append((executed_qty, executed_price))
        # episode lists
        self.episode_executed_orders.append(order)
        self.parsed_episode_executed_orders.append((executed_qty, executed_price))
        # update order status dictionnary
        # test if it was mkt order and first execution received from it
        try:
            self.order_status[order_id]
            flag = True
        except KeyError:
            flag = False

        if flag:
            self.order_status[order_id]["executed_qty"] += executed_qty
            self.order_status[order_id]["active_qty"] -= executed_qty
            if self.order_status[order_id]["active_qty"] <= 0:
                self.order_status[order_id]["status"] = "executed"
        else:
            self.order_status[order_id] = {
                "status": "mkt_immediately_filled",
                "order": order,
                "active_qty": 0,
                "executed_qty": executed_qty,
                "cancelled_qty": 0,
            }

    def order_accepted(self, order: Order) -> None:
        super().order_accepted(order)
        # update order status dictionnary
        self.order_status[order.order_id] = {
            "status": "active",
            "order": order,
            "active_qty": order.quantity,
            "executed_qty": 0,
            "cancelled_qty": 0,
        }

    def order_cancelled(self, order: Order) -> None:
        # This method in the parent does not require an exchange_id, so it is okay.
        super().order_cancelled(order)
        order_id = order.order_id
        if order_id in self.order_status:
            original_order_qty = self.order_status[order_id]["order"].quantity
            executed_qty = self.order_status[order_id]["executed_qty"]

            self.order_status[order_id]["status"] = "cancelled"
            self.order_status[order_id]["cancelled_qty"] = original_order_qty - executed_qty

    def new_inter_wakeup_reset(self) -> None:
        self.inter_wakeup_executed_orders = (
            []
        )  # list of executed orders between steps - is reset at every step
        self.parsed_inter_wakeup_executed_orders = []  # just tuple (price, qty)

    def act(self, raw_state):
        # used by the background agent
        raise NotImplementedError

    def new_step_reset(self) -> None:
        self.inter_wakeup_executed_orders = []
        self.parsed_inter_wakeup_executed_orders = []

