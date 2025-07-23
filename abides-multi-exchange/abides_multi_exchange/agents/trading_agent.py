import logging
import sys
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import fmt_ts

from abides_markets.messages.market import (
    MarketClosePriceRequestMsg,
    MarketClosePriceMsg,
    MarketClosedMsg,
    MarketHoursRequestMsg,
    MarketHoursMsg,
)
from abides_markets.messages.marketdata import MarketDataSubReqMsg, MarketDataMsg, L2DataMsg
from abides_markets.messages.order import (
    LimitOrderMsg,
    MarketOrderMsg,
    PartialCancelOrderMsg,
    CancelOrderMsg,
    ModifyOrderMsg,
    ReplaceOrderMsg,
)
from abides_markets.messages.orderbook import (
    OrderAcceptedMsg,
    OrderExecutedMsg,
    OrderCancelledMsg,
    OrderPartialCancelledMsg,
    OrderModifiedMsg,
    OrderReplacedMsg,
)
from abides_markets.messages.query import (
    QueryLastTradeMsg,
    QueryLastTradeResponseMsg,
    QuerySpreadMsg,
    QuerySpreadResponseMsg,
    QueryOrderStreamMsg,
    QueryOrderStreamResponseMsg,
    QueryTransactedVolMsg,
    QueryTransactedVolResponseMsg,
)
from abides_markets.orders import Order, LimitOrder, MarketOrder, Side
from abides_markets.agents.financial_agent import FinancialAgent
from abides_markets.agents.exchange_agent import ExchangeAgent
from abides_multi_exchange.messages import CompleteTransferMsg

logger = logging.getLogger(__name__)


class TradingAgent(FinancialAgent):
    """
    The TradingAgent class (via FinancialAgent, via Agent) is intended as the
    base class for all trading agents (i.e. not things like exchanges) in a
    market simulation.

    It handles a lot of messaging (inbound and outbound) and state maintenance
    automatically, so subclasses can focus just on implementing a strategy without
    too much bookkeeping.
    """

    def __init__(
            self,
            id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            starting_cash: int = 100000,
            log_orders: bool = False,
            exchange_id: Optional[Union[int, List[int]]] = None,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state)

        # We don't yet know when the exchange opens or closes.
        # ME: changed mkt_open and close to dictionary with index exchange_id
        self.mkt_open: Dict[int, NanosecondTime] = {}
        self.mkt_close: Dict[int, NanosecondTime] = {}

        # Log order activity?
        self.log_orders: bool = log_orders

        # Log all activity to file?
        if log_orders is None:
            self.log_orders = False
            self.log_to_file = False

        # Store starting_cash in case we want to refer to it for performance stats.
        # It should NOT be modified.  Use the 'CASH' key in self.holdings.
        # 'CASH' is always in cents!  Note that agents are limited by their starting
        # cash, currently without leverage.  Taking short positions is permitted,
        # but does NOT increase the amount of at-risk capital allowed.
        self.starting_cash: int = starting_cash

        # TradingAgent has constants to support simulated market orders.
        self.MKT_BUY = sys.maxsize
        self.MKT_SELL = 0

        # The base TradingAgent will track its holdings and outstanding orders.
        # Holdings is a dictionary of symbol -> shares.  CASH is a special symbol
        # worth one cent per share.  Orders is a dictionary of active, open orders
        # (not cancelled, not fully executed) keyed by order_id.
        # ME: with withdrawal fees (gas costs)
        self.holdings_by_exchange: Dict[int, Dict[str, int]] = {}
        self.cash: int = starting_cash

        # Multiple exchanges: self.orders will now store a tuple: (Order, exchange_id)
        self.orders: Dict[int, Tuple[Order, int]] = {}

        # The base TradingAgent also tracks last known prices for every symbol
        # for which it has received as QUERY_LAST_TRADE message.  Subclass
        # agents may use or ignore this as they wish.  Note that the subclass
        # agent must request pricing when it wants it.  This agent does NOT
        # automatically generate such requests, though it has a helper function
        # that can be used to make it happen.
        # ME: these will now be dictionaries keyed by exchange_id
        self.last_trade: Dict[int, Dict[str, int]] = {}

        # used in subscription mode to record the timestamp for which the data was current in the ExchangeAgent
        self.exchange_ts: Dict[str, NanosecondTime] = {}

        # When a last trade price comes in after market close, the trading agent
        # automatically records it as the daily close price for a symbol.
        self.daily_close_price: Dict[str, int] = {}

        self.nav_diff: int = 0
        self.basket_size: int = 0

        # The agent remembers the last known bids and asks (with variable depth,
        # showing only aggregate volume at each price level) when it receives
        # a response to QUERY_SPREAD.
        # ME: This is now a dictionaries keyed by exchange_id
        self.known_bids: Dict[int, Dict] = {}
        self.known_asks: Dict[int, Dict] = {}

        # The agent remembers the order history communicated by the exchange
        # when such is requested by an agent (for example, a heuristic belief
        # learning agent).
        # ME: This is now a dictionaries keyed by exchange_id
        self.stream_history: Dict[int, Dict[str, Any]] = {}

        # The agent records the total transacted volume in the exchange for a given symbol and lookback period
        self.transacted_volume: Dict = {}

        # Each agent can choose to log the orders executed
        self.executed_orders: List = []

        # For special logging at the first moment the simulator kernel begins
        # running (which is well after agent init), it is useful to keep a simple
        # boolean flag.
        self.first_wake: bool = True

        # Remember whether we have already passed the exchange close time, as far
        # as we know.
        # ME: This is now a dictionaries keyed by exchange_id
        self.mkt_closed: Dict[int, bool] = {}

        self.book: Dict[int, str] = {}

        # Transfer delay specification
        self.min_transfer_delay_minutes: int = 5
        self.max_transfer_delay_minutes: int = 70
        self.transfer_delay_alpha: int = 5
        self.transfer_delay_beta: int = 14.5

        # Store the exchange IDs
        if exchange_id is None:
            self.exchange_ids: List[int] = []
        elif isinstance(exchange_id, int):
            self.exchange_ids: List[int] = [exchange_id]
        else:
            self.exchange_ids: List[int] = exchange_id

    # Simulation lifecycle messages.

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        """
        Arguments:
            start_time: The time that the simulation started.
        """

        assert self.kernel is not None

        # self.kernel is set in Agent.kernel_initializing()
        self.logEvent("STARTING_CASH", self.starting_cash, True)

        # Find exchanges with which we can place orders.
        if not self.exchange_ids:
            self.exchange_ids = [a for a in self.kernel.find_agents_by_type(ExchangeAgent)]

        for ex_id in self.exchange_ids:
            logger.debug(
                f"Agent {self.id} will communicate with Exchange ID: {ex_id}"
            )
            self.holdings_by_exchange[ex_id] = {}
            self.known_bids[ex_id] = {}
            self.known_asks[ex_id] = {}
            self.last_trade[ex_id] = {}
            self.stream_history[ex_id] = {}
            self.transacted_volume[ex_id] = {}
            self.mkt_closed[ex_id] = False

        # Request a wake-up call as in the base Agent.
        super().kernel_starting(start_time)

        # Retrieve the global fee config from the kernel's custom properties
        self.withdrawal_fees = getattr(self.kernel, 'withdrawal_fees', {})


    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        assert self.kernel is not None

        # Print end of day holdings.
        self.logEvent(
            "FINAL_HOLDINGS", self.fmt_holdings(self.holdings_by_exchange), deepcopy_event=False
        )
        self.logEvent("FINAL_CASH_POSITION", self.cash, True)

        final_portfolio_value = self.mark_to_market(self.holdings_by_exchange, self.cash, self.withdrawal_fees)

        self.logEvent("ENDING_CASH", final_portfolio_value, True)
        logger.debug(
            f"Final holdings for {self.name}: {self.fmt_holdings(self.holdings_by_exchange)}. Marked to market: {final_portfolio_value}"
        )

        # Record final results for presentation/debugging.  This is an ugly way
        # to do this, but it is useful for now.
        mytype = self.type
        gain = final_portfolio_value - self.starting_cash

        if mytype in self.kernel.mean_result_by_agent_type:
            self.kernel.mean_result_by_agent_type[mytype] += gain
            self.kernel.agent_count_by_type[mytype] += 1
        else:
            self.kernel.mean_result_by_agent_type[mytype] = gain
            self.kernel.agent_count_by_type[mytype] = 1

    # Simulation participation messages.

    def wakeup(self, current_time: NanosecondTime) -> bool:
        """
        Arguments:
            current_time: The time that this agent was woken up by the kernel.

        Returns:
            For the sake of subclasses, TradingAgent now returns a boolean
            indicating whether the agent is "ready to trade" -- has it received
            the market open and closed times, and is the market not already closed.
        """

        super().wakeup(current_time)

        if self.first_wake:
            # Log initial holdings.
            self.logEvent("HOLDINGS_UPDATED", self.fmt_holdings(self.holdings_by_exchange))
            self.first_wake = False

            # Tell the exchanges we want to be sent the final prices when the market closes.
            for exchange_id in self.exchange_ids:
                self.send_message(exchange_id, MarketClosePriceRequestMsg())

        for exchange_id in self.exchange_ids:
            if exchange_id not in self.mkt_open:
                # Ask our exchange when it opens and closes.
                self.send_message(exchange_id, MarketHoursRequestMsg())

        # Return true if all exchanges are open and not closed.
        # Update (remove comment when obsolete): returns true when there is at least one open exchange
        all_closed = len(self.mkt_closed) > 0 and all(self.mkt_closed.get(eid, False) for eid in self.exchange_ids)
        return all(eid in self.mkt_open and eid in self.mkt_close for eid in self.exchange_ids) and not all_closed

    def request_data_subscription(
            self, exchange_id: int, subscription_message: MarketDataSubReqMsg
    ) -> None:
        """
        Used by any Trading Agent subclass to create a subscription to market data from
        the Exchange Agent.

        Arguments:
            exchange_id: The ID of the exchange to subscribe to.
            subscription_message: An instance of a MarketDataSubReqMessage.
        """

        subscription_message.cancel = False

        self.send_message(recipient_id=exchange_id, message=subscription_message)

    def cancel_data_subscription(
            self, exchange_id: int, subscription_message: MarketDataSubReqMsg
    ) -> None:
        """
        Used by any Trading Agent subclass to cancel subscription to market data from
        the Exchange Agent.

        Arguments:
            exchange_id: The ID of the exchange to unsubscribe from.
            subscription_message: An instance of a MarketDataSubReqMessage.
        """

        subscription_message.cancel = True

        self.send_message(recipient_id=exchange_id, message=subscription_message)

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """
        Arguments:
            current_time: The time that this agent received the message.
            sender_id: The ID of the agent who sent the message.
            message: The message contents.
        """

        assert self.kernel is not None

        super().receive_message(current_time, sender_id, message)

        # Do we know the market hours for this exchange?
        had_mkt_hours = sender_id in self.mkt_open and sender_id in self.mkt_close

        # Record market open or close times.
        if isinstance(message, MarketHoursMsg):
            self.mkt_open[sender_id] = message.mkt_open
            self.mkt_close[sender_id] = message.mkt_close

            logger.debug(f"Recorded market open for exchange {sender_id}: {fmt_ts(self.mkt_open[sender_id])}")
            logger.debug(f"Recorded market close for exchange {sender_id}: {fmt_ts(self.mkt_close[sender_id])}")

        elif isinstance(message, MarketClosePriceMsg):
            # Update our local last trade prices with the accurate last trade prices from
            # the exchange so we can accurately calculate our mark-to-market values.
            for symbol, close_price in message.close_prices.items():
                self.last_trade[sender_id][symbol] = close_price

        elif isinstance(message, MarketClosedMsg):
            # We've tried to ask the exchange for something after it closed.  Remember this
            # so we stop asking for things that can't happen.
            self.market_closed(sender_id)

        elif isinstance(message, CompleteTransferMsg):
            # Call the _complete transfer method
            self._complete_transfer(
                to_exchange=message.to_exchange,
                symbol=message.symbol,
                size=message.size
            )
            return

        elif isinstance(message, OrderExecutedMsg):
            # Call the order_executed method, which subclasses should extend.  This parent
            # class could implement default "portfolio tracking" or "returns tracking"
            # behavior.
            self.order_executed(sender_id, message.order)

        elif isinstance(message, OrderAcceptedMsg):
            # Call the order_accepted method, which subclasses should extend.
            self.order_accepted(message.order)

        elif isinstance(message, OrderCancelledMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_cancelled(message.order)

        elif isinstance(message, OrderPartialCancelledMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_partial_cancelled(message.new_order)

        elif isinstance(message, OrderModifiedMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_modified(message.new_order)

        elif isinstance(message, OrderReplacedMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_replaced(message.old_order, message.new_order)

        elif isinstance(message, QueryLastTradeResponseMsg):
            # Call the query_last_trade method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_last_trade(sender_id, message.symbol, message.last_trade)

        elif isinstance(message, QuerySpreadResponseMsg):
            # Call the query_spread method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_spread(
                sender_id, message.symbol, message.last_trade, message.bids, message.asks, ""
            )

        elif isinstance(message, QueryOrderStreamResponseMsg):
            # Call the query_order_stream method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_order_stream(sender_id, message.symbol, message.orders)

        elif isinstance(message, QueryTransactedVolResponseMsg):
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_transacted_volume(
                sender_id, message.symbol, message.bid_volume, message.ask_volume
            )

        elif isinstance(message, MarketDataMsg):
            self.handle_market_data(sender_id, message)

        # Now do we know the market hours?
        have_mkt_hours = sender_id in self.mkt_open and sender_id in self.mkt_close

        # Once we know the market open and close times, schedule a wakeup call for market open.
        # Only do this once, when we first have both items.
        if have_mkt_hours and not had_mkt_hours:
            # Agents are asked to generate a wake offset from the market open time.  We structure
            # this as a subclass request so each agent can supply an appropriate offset relative
            # to its trading frequency.
            ns_offset = self.get_wake_frequency()

            self.set_wakeup(self.mkt_open[sender_id] + ns_offset)

    def get_last_trade(self, exchange_id: int, symbol: str) -> None:
        """
        Used by any Trading Agent subclass to query the last trade price for a symbol.

        This activity is not logged.

        Arguments:
            exchange_id: The ID of the exchange to query.
            symbol: The symbol to query.
        """

        self.send_message(exchange_id, QueryLastTradeMsg(symbol))

    def get_current_spread(self, exchange_id: int, symbol: str, depth: int = 1) -> None:
        """
        Used by any Trading Agent subclass to query the current spread for a symbol.

        This activity is not logged.

        Arguments:
            exchange_id: The ID of the exchange to query.
            symbol: The symbol to query.
            depth:
        """

        self.send_message(exchange_id, QuerySpreadMsg(symbol, depth))

    def get_order_stream(self, exchange_id: int, symbol: str, length: int = 1) -> None:
        """
        Used by any Trading Agent subclass to query the recent order s  tream for a symbol.

        Arguments:
            exchange_id: The ID of the exchange to query.
            symbol: The symbol to query.
            length:
        """

        self.send_message(exchange_id, QueryOrderStreamMsg(symbol, length))

    def get_transacted_volume(
            self, exchange_id: int, symbol: str, lookback_period: str = "10min"
    ) -> None:
        """
        Used by any trading agent subclass to query the total transacted volume in a
        given lookback period.

        Arguments:
            exchange_id: The ID of the exchange to query.
            symbol: The symbol to query.
            lookback_period: The length of time to consider when calculating the volume.
        """

        self.send_message(
            exchange_id, QueryTransactedVolMsg(symbol, lookback_period)
        )

    def create_limit_order(
        self,
        exchange_id: int,
        symbol: str,
        quantity: int,
        side: Side,
        limit_price: int,
        order_id: Optional[int] = None,
        is_hidden: bool = False,
        is_price_to_comply: bool = False,
        insert_by_id: bool = False,
        is_post_only: bool = False,
        ignore_risk: bool = True,
        tag: Any = None,
    ) -> Optional[LimitOrder]:
        """
        Used by any Trading Agent subclass to create a limit order.

        Arguments:
            symbol: A valid symbol.
            quantity: Positive share quantity.
            side: Side.BID or Side.ASK.
            limit_price: Price in cents.
            order_id: An optional order id (otherwise global autoincrement is used).
            is_hidden:
            is_price_to_comply:
            insert_by_id:
            is_post_only:
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = LimitOrder(
            agent_id=self.id,
            time_placed=self.current_time,
            symbol=symbol,
            quantity=quantity,
            side=side,
            limit_price=limit_price,
            is_hidden=is_hidden,
            is_price_to_comply=is_price_to_comply,
            insert_by_id=insert_by_id,
            is_post_only=is_post_only,
            order_id=order_id,
            tag=tag,
        )

        if quantity > 0:
            # Test if this order can be permitted given our at-risk limits.
            new_holdings = deepcopy(self.holdings_by_exchange)
            holdings_at_exchange = new_holdings.setdefault(exchange_id, {})

            q = order.quantity if order.side.is_bid() else -order.quantity

            holdings_at_exchange[symbol] = holdings_at_exchange.get(symbol, 0) + q

            # If at_risk is lower, always allow.  Otherwise, new_at_risk must be below starting cash.
            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.mark_to_market(self.holdings_by_exchange, self.cash, self.withdrawal_fees) - self.cash
                new_at_risk = self.mark_to_market(new_holdings, self.cash, self.withdrawal_fees) - self.cash

                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    logger.debug(
                        f"TradingAgent ignored limit order due to at-risk constraints: {order}\n{self.fmt_holdings(self.holdings_by_exchange)}"
                    )
                    return None

            return order

        else:
            warnings.warn(f"TradingAgent ignored limit order of quantity zero or below: {order}")
            return None

    def place_limit_order(
            self,
            exchange_id: int,
            symbol: str,
            quantity: int,
            side: Side,
            limit_price: int,
            order_id: Optional[int] = None,
            is_hidden: bool = False,
            is_price_to_comply: bool = False,
            insert_by_id: bool = False,
            is_post_only: bool = False,
            ignore_risk: bool = True,
            tag: Any = None,
    ) -> None:
        """
        Used by any Trading Agent subclass to place a limit order.

        Arguments:
            exchange_id: The ID of the exchange to send the order to.
            symbol: A valid symbol.
            quantity: Positive share quantity.
            side: Side.BID or Side.ASK.
            limit_price: Price in cents.
            order_id: An optional order id (otherwise global autoincrement is used).
            is_hidden:
            is_price_to_comply:
            insert_by_id:
            is_post_only:
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = self.create_limit_order(
            exchange_id,
            symbol,
            quantity,
            side,
            limit_price,
            order_id,
            is_hidden,
            is_price_to_comply,
            insert_by_id,
            is_post_only,
            ignore_risk,
            tag,
        )

        if order is not None:
            self.orders[order.order_id] = (deepcopy(order), exchange_id)
            self.send_message(exchange_id, LimitOrderMsg(order))

            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

    def place_market_order(
            self,
            exchange_id: int,
            symbol: str,
            quantity: int,
            side: Side,
            order_id: Optional[int] = None,
            ignore_risk: bool = True,
            tag: Any = None,
    ) -> None:
        """
        Used by any Trading Agent subclass to place a market order.

        The market order is created as multiple limit orders crossing the spread
        walking the book until all the quantities are matched.

        Arguments:
            exchange_id: The ID of the exchange to send the order to.
            symbol: Name of the stock traded.
            quantity: Order quantity.
            side: Side.BID or Side.ASK.
            order_id: Order ID for market replay.
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = MarketOrder(
            self.id, self.current_time, symbol, quantity, side, order_id, tag
        )
        if quantity > 0:
            # compute new holdings
            new_holdings = deepcopy(self.holdings_by_exchange)
            holdings_at_exchange = new_holdings.setdefault(exchange_id, {})
            q = order.quantity if order.side.is_bid() else -order.quantity

            holdings_at_exchange[symbol] = holdings_at_exchange.get(symbol, 0) + q

            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.mark_to_market(self.holdings_by_exchange, self.cash, self.withdrawal_fees) - self.cash
                new_at_risk = self.mark_to_market(new_holdings, self.cash, self.withdrawal_fees) - self.cash

                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    logger.debug(
                        f"TradingAgent ignored market order due to at-risk constraints: {order}\n{self.fmt_holdings(self.holdings_by_exchange)}"
                    )
                    return
            self.orders[order.order_id] = (deepcopy(order), exchange_id)
            self.send_message(exchange_id, MarketOrderMsg(order))
            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

        else:
            warnings.warn(f"TradingAgent ignored market order of quantity zero or less: {order}")
            return None

    def place_multiple_orders(
            self, exchange_id: int, orders: List[Union[LimitOrder, MarketOrder]]
    ) -> None:
        """
        Used by any Trading Agent subclass to place multiple orders at the same time.

        Arguments:
            exchange_id: The ID of the exchange to send the orders to.
            orders: A list of Orders to place with the exchange as a single batch.
        """

        messages = []

        for order in orders:
            if isinstance(order, LimitOrder):
                messages.append(LimitOrderMsg(order))
            elif isinstance(order, MarketOrder):
                messages.append(MarketOrderMsg(order))
            else:
                raise Exception("Expected LimitOrder or MarketOrder")

            # Copy the intended order for logging, so any changes made to it elsewhere
            # don't retroactively alter our "as placed" log of the order.  Eventually
            # it might be nice to make the whole history of the order into transaction
            # objects inside the order (we're halfway there) so there CAN be just a single
            # object per order, that never alters its original state, and eliminate all
            # these copies.
            self.orders[order.order_id] = (deepcopy(order), exchange_id)

            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

        if len(messages) > 0:
            self.send_message_batch(exchange_id, messages)

    def cancel_order(
            self, exchange_id: int, order: LimitOrder, tag: Optional[str] = None, metadata: dict = {}
    ) -> None:
        """
        Used by derived classes of TradingAgent to cancel a limit order.

        The order must currently appear in the agent's open orders list.

        Arguments:
            exchange_id: The ID of the exchange to send the cancellation to.
            order: The limit order to cancel.
            tag:
            metadata:
        """

        if isinstance(order, LimitOrder):
            self.send_message(exchange_id, CancelOrderMsg(order, tag, metadata))
            if self.log_orders:
                self.logEvent("CANCEL_SUBMITTED", order.to_dict(), deepcopy_event=False)
        else:
            warnings.warn(f"Order {order} of type, {type(order)} cannot be cancelled")

    def cancel_all_orders(self):
        """
        Cancels all current limit orders held by this agent across all exchanges.
        """
        # Iterate through the stored order tuples.
        for order, exchange_id in self.orders.values():
            if isinstance(order, LimitOrder):
                # Use the stored exchange_id to cancel the order.
                self.cancel_order(exchange_id, order)

    def partial_cancel_order(
            self,
            exchange_id: int,
            order: LimitOrder,
            quantity: int,
            tag: Optional[str] = None,
            metadata: dict = {},
    ) -> None:
        """
        Used by any Trading Agent subclass to partially cancel an existing limit order.

        Arguments:
            exchange_id: The ID of the exchange where the order resides.
            order: The limit order to partially cancel.
            quantity:
            tag:
            metadata:
        """
        self.send_message(
            exchange_id, PartialCancelOrderMsg(order, quantity, tag, metadata)
        )

        if self.log_orders:
            self.logEvent("CANCEL_PARTIAL_ORDER", order.to_dict(), deepcopy_event=False)

    def modify_order(self, exchange_id: int, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to modify any existing limit order.

        The order must currently appear in the agent's open orders list.  Some
        additional tests might be useful here to ensure the old and new orders are
        the same in some way.

        Arguments:
            exchange_id: The ID of the exchange where the order resides.
            order: The existing limit order.
            new_order: The limit order to update the existing order with.
        """
        self.send_message(exchange_id, ModifyOrderMsg(order, new_order))

        if self.log_orders:
            self.logEvent("MODIFY_ORDER", order.to_dict(), deepcopy_event=False)

    def replace_order(self, exchange_id: int, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to replace any existing limit order.

        Arguments:
            exchange_id: The ID of the exchange where the order resides.
            order: The existing limit order.
            new_order: The new limit order to replace the existing order with.
        """
        self.send_message(exchange_id, ReplaceOrderMsg(self.id, order, new_order))

        if self.log_orders:
            self.logEvent("REPLACE_ORDER", order.to_dict(), deepcopy_event=False)

    def order_executed(self, exchange_id: int, order: Order) -> None:
        """
        Handles OrderExecuted messages from an exchange agent.

        Subclasses may wish to extend, but should still call parent method for basic
        portfolio/returns tracking.

        Arguments:
            order: The order that has been executed by the exchange.
        """
        # print(f"{self.id}:{self.name} Received notification of execution for {order.order_id} from Exchange {exchange_id} ")
        logger.debug(f"Received notification of execution for {order.order_id} from Exchange {exchange_id} ")

        if self.log_orders:
            self.logEvent("ORDER_EXECUTED", order.to_dict(), deepcopy_event=False)

        # At the very least, we must update CASH and holdings at execution time.
        # Update holdings (this logic is exchange-agnostic).
        qty = order.quantity if order.side.is_bid() else -1 * order.quantity
        sym = order.symbol

        holdings_on_this_exchange = self.holdings_by_exchange[exchange_id]
        holdings_on_this_exchange[sym] = holdings_on_this_exchange.get(sym, 0) + qty


        if self.holdings_by_exchange[exchange_id][sym] == 0:
            del self.holdings_by_exchange[exchange_id][sym]

        # As with everything else, CASH holdings are in CENTS.
        self.cash -= qty * order.fill_price

        # If this original order is now fully executed, remove it from the open orders list.
        # Otherwise, decrement by the quantity filled just now.  It is _possible_ that due
        # to timing issues, it might not be in the order list (i.e. we issued a cancellation
        # but it was executed first, or something).
        if order.order_id in self.orders:
            # Retrieve the tuple (original_order, exchange_id).
            original_order, _ = self.orders[order.order_id]

            if order.quantity >= original_order.quantity:
                del self.orders[order.order_id]
            else:
                original_order.quantity -= order.quantity
        else:
            warnings.warn(f"Execution received for order not in orders list: {order}")

        logger.debug(f"After order execution, agent open orders: {self.orders}")
        logger.debug(f"Holdings at {exchange_id}: {self.holdings_by_exchange[exchange_id]}")
        logger.debug(f"Total Cash: {self.cash}")

        self.logEvent("HOLDINGS_UPDATED", self.fmt_holdings(self.holdings_by_exchange))

    def order_accepted(self, order: LimitOrder) -> None:
        """
        Handles OrderAccepted messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been accepted from the exchange.
        """
        logger.debug(f"Received notification of acceptance for: {order}")
        if self.log_orders:
            self.logEvent("ORDER_ACCEPTED", order.to_dict(), deepcopy_event=False)

        # We may later wish to add a status to the open orders so an agent can tell whether
        # a given order has been accepted or not (instead of needing to override this method).

    def order_cancelled(self, order: LimitOrder) -> None:
        """
        Handles OrderCancelled messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been cancelled by the exchange.
        """

        logger.debug(f"Received notification of cancellation for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_CANCELLED", order.to_dict(), deepcopy_event=False)

        # Remove the cancelled order from the open orders list.  We may of course wish to have
        # additional logic here later, so agents can easily "look for" cancelled orders.  Of
        # course they can just override this method.
        if order.order_id in self.orders:
            del self.orders[order.order_id]
        else:
            warnings.warn(f"Cancellation received for order not in orders list: {order}")

    def order_partial_cancelled(self, order: LimitOrder) -> None:
        """
        Handles OrderCancelled messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been partially cancelled by the exchange.
        """

        logger.debug(f"Received notification of partial cancellation for: {order}")

        if self.log_orders:
            self.logEvent("PARTIAL_CANCELLED", order.to_dict())
        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if order.order_id in self.orders:
            # Update the order part of the tuple. Keep the exchange_id.
            _, exchange_id = self.orders[order.order_id]
            self.orders[order.order_id] = (order, exchange_id)
        else:
            warnings.warn(f"Partial cancellation for order not in orders list: {order}")

        logger.debug(
            f"After order partial cancellation, agent open orders: {self.orders}"
        )

        self.logEvent("HOLDINGS_UPDATED", self.fmt_holdings(self.holdings_by_exchange))
    def order_modified(self, order: LimitOrder) -> None:
        """
        Handles OrderModified messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been modified at the exchange.
        """

        logger.debug(f"Received notification of modification for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_MODIFIED", order.to_dict())

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if order.order_id in self.orders:
            _, exchange_id = self.orders[order.order_id]
            self.orders[order.order_id] = (order, exchange_id)

        else:
            warnings.warn(f"Modification received for order not in orders list: {order}")

        logger.debug(f"After order modification, agent open orders: {self.orders}")

        self.logEvent("HOLDINGS_UPDATED", self.fmt_holdings(self.holdings_by_exchange))

    def order_replaced(self, old_order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Handles OrderReplaced messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been modified at the exchange.
        """

        logger.debug(f"Received notification of replacement for: {old_order}")

        if self.log_orders:
            self.logEvent("ORDER_REPLACED", old_order.to_dict())

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if old_order.order_id in self.orders:
            # ME: Get the original exchange_id and assign it to the new order
            _, exchange_id = self.orders[old_order.order_id]
            del self.orders[old_order.order_id]
            self.orders[new_order.order_id] = (new_order, exchange_id)

        else:
            warnings.warn(
                f"Replacement received for order not in orders list: {old_order}"
            )

        logger.debug(f"After order replacement, agent open orders: {self.orders}")

        # After execution, log holdings.
        self.logEvent("HOLDINGS_UPDATED", self.fmt_holdings(self.holdings_by_exchange))

    def transfer_asset(self, symbol, from_exchange, to_exchange, amount, withdrawal_fee):
        """
        Simulates withdrawing an asset from one exchange and depositing to another.
        """
        # TODO: Need to add logEvents and logger codes to this method
        # Check if we have enough to transfer
        if self.holdings_by_exchange[from_exchange].get(symbol, 0) >= amount:

            # Debit from the source exchange
            self.holdings_by_exchange[from_exchange][symbol] -= amount

            # Apply the fixed withdrawal fee
            amount_to_receive = amount - withdrawal_fee

            # Credit to the destination exchange
            # TODO: This could be delayed to simulate blockchain confirmation time
            if amount_to_receive > 0:
                self.holdings_by_exchange[to_exchange][symbol] = \
                    self.holdings_by_exchange[to_exchange].get(symbol, 0) + amount_to_receive

    def _get_random_transfer_delay(self) -> NanosecondTime:
        """
        Generates a random transfer delay skewed by a Beta distribution.
        """
        sample = self.random_state.beta(self.transfer_delay_alpha, self.transfer_delay_beta)

        delay_range = self.max_transfer_delay_minutes - self.min_transfer_delay_minutes
        delay_minutes = self.min_transfer_delay_minutes + sample * delay_range
        delay_ns = int(delay_minutes * 60 * 1_000_000_000)

        logger.debug(f"Generated random transfer delay of ~{delay_minutes:.2f} minutes ({delay_ns} ns)")
        # Convert to nanoseconds and return as an integer
        return int(delay_minutes * 60 * 1_000_000_000)

    def _complete_transfer(self, to_exchange: int, symbol: str, size: int) -> None:
        """
        Callback function executed by the kernel after the transfer_delay.
        It deposits the in-transit assets into the destination exchange.
        """
        # Add the transferred assets to the destination exchange's holdings
        if to_exchange not in self.holdings_by_exchange:
            self.holdings_by_exchange[to_exchange] = {}
        if symbol not in self.holdings_by_exchange[to_exchange]:
            self.holdings_by_exchange[to_exchange][symbol] = 0

        self.holdings_by_exchange[to_exchange][symbol] += size

        self.logEvent("TRANSFER_COMPLETE", f"{size} shares of {symbol} arrived at Exchange {to_exchange}")
        print(f"DEBUG ({self.name}): Transfer complete. {size} shares of {symbol} "
          f"arrived at Exchange {to_exchange}.")


    def market_closed(self, exchange_id: int) -> None:
        """
        Handles MarketClosedMsg messages from a specific exchange agent.

        Subclasses may wish to extend.
        """

        logger.debug(f"Received notification of market closure from exchange {exchange_id}.")

        # Log this activity.
        self.logEvent("MKT_CLOSED", exchange_id)

        # Remember that this has happened.
        self.mkt_closed[exchange_id] = True

    def query_last_trade(self, exchange_id: int, symbol: str, price: int) -> None:
        """
        Handles QueryLastTradeResponseMsg messages from an exchange agent.

        Arguments:
            exchange_id: The ID of the exchange that sent the response.
            symbol: The symbol that was queried.
            price: The price at which the last trade was executed at.
        """

        # Store the last trade price per exchange and per symbol.
        if exchange_id not in self.last_trade:
            self.last_trade[exchange_id] = {}
        self.last_trade[exchange_id][symbol] = price

        logger.debug(f"Received last trade of {price} for {symbol} from exchange {exchange_id}.")

        if self.mkt_closed.get(exchange_id, False):
            # The daily close price can be handled per symbol, assuming it's the last
            # price seen from any exchange before simulation ends.
            self.daily_close_price[symbol] = price
            logger.debug(f"Received daily close price of {price} for {symbol} for exchange {exchange_id}.")

    def query_spread(
            self,
            exchange_id: int,
            symbol: str,
            price: int,
            bids: List[List[Tuple[int, int]]],
            asks: List[List[Tuple[int, int]]],
            book: str,
    ) -> None:
        """
        Handles QuerySpreadResponseMsg messages from an exchange agent.

        Arguments:
            exchange_id: The ID of the exchange that sent the response.
            symbol: The symbol that was queried.
            price:
            bids:
            asks:
            book:
        """

        # The spread message now also includes last price for free.
        self.query_last_trade(exchange_id, symbol, price)

        self.known_bids[exchange_id][symbol] = bids
        self.known_asks[exchange_id][symbol] = asks

        if bids:
            best_bid, best_bid_qty = (bids[0][0], bids[0][1])
        else:
            best_bid, best_bid_qty = ("No bids", 0)

        if asks:
            best_ask, best_ask_qty = (asks[0][0], asks[0][1])
        else:
            best_ask, best_ask_qty = ("No asks", 0)

        logger.debug(
            f"Received spread of {best_bid_qty} @ {best_bid} / {best_ask_qty} @ {best_ask} for {symbol} from exchange {exchange_id}"
        )

        self.logEvent("BID_DEPTH", bids)
        self.logEvent("ASK_DEPTH", asks)
        self.logEvent(
            "IMBALANCE", [sum([x[1] for x in bids]), sum([x[1] for x in asks])]
        )

        self.book[exchange_id] = book

    def handle_market_data(self, exchange_id: int, message: MarketDataMsg) -> None:
        """
        Handles Market Data messages for agents using subscription mechanism.

        Arguments:
            exchange_id: The id of the exchange that sent the message.
            message: The market data message,
        """

        if isinstance(message, L2DataMsg):
            symbol = message.symbol
            self.known_asks[exchange_id][symbol] = message.asks
            self.known_bids[exchange_id][symbol] = message.bids
            self.last_trade[exchange_id][symbol] = message.last_transaction
            self.exchange_ts[symbol] = message.exchange_ts

    def query_order_stream(self, exchange_id: int, symbol: str, orders) -> None:
        """
        Handles QueryOrderStreamResponseMsg messages from an exchange agent.

        It is up to the requesting agent to do something with the data, which is a list
        of dictionaries keyed by order id. The list index is 0 for orders since the most
        recent trade, 1 for orders that led up to the most recent trade, and so on.
        Agents are not given index 0 (orders more recent than the last trade).

        Arguments:
            exchange_id: The ID of the exchange that sent the data
            symbol: The symbol that was queried.
            orders:
        """
        if exchange_id not in self.stream_history:
            self.stream_history[exchange_id] = {}
        self.stream_history[exchange_id][symbol] = orders


    def query_transacted_volume(
        self, exchange_id: int, symbol: str, bid_volume: int, ask_volume: int
    ) -> None:
        """
        Handles the QueryTransactedVolResponseMsg messages from the exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            bid_vol: The volume that has transacted on the bid side for the queried period.
            ask_vol: The volume that has transacted on the ask side for the queried period.
        """
        if exchange_id not in self.transacted_volume:
            self.transacted_volume[exchange_id] = {}
        self.transacted_volume[exchange_id][symbol] = (bid_volume, ask_volume)


    # Utility functions that perform calculations from available knowledge, but implement no
    # particular strategy.

    def get_known_bid_ask(self, exchange_id: int, symbol: str, best: bool = True):
        """
        Extract the current known bid and asks.

        This does NOT request new information.

        Arguments:
            exchange_id: The ID of the exchange to query
            symbol: The symbol to query.
            best: whether to return only the best bid/ask or the full book side
        """

        bids = self.known_bids.get(exchange_id, {}).get(symbol,[])
        asks = self.known_asks.get(exchange_id, {}).get(symbol,[])
        if best:
            bid = bids[0][0] if bids else None
            ask = asks[0][0] if asks else None
            bid_vol = bids[0][1] if bids else 0
            ask_vol = asks[0][1] if asks else 0
            return bid, bid_vol, ask, ask_vol
        else:
            return bids, asks

    def get_known_liquidity(self, exchange_id: int, symbol: str, within: float = 0.00) -> Tuple[int, int]:
        """
        Extract the current bid and ask liquidity within a certain proportion of the
        inside bid and ask.  (i.e. within=0.01 means to report total BID shares
        within 1% of the best bid price, and total ASK shares within 1% of the best
        ask price)

        Arguments:
            symbol: The symbol to query.
            within:

        Returns:
            (bid_liquidity, ask_liquidity).  Note that this is from the order book
            perspective, not the agent perspective.  (The agent would be selling into
            the bid liquidity, etc.)
        """

        bids, asks = self.get_known_bid_ask(exchange_id, symbol, best=False)

        bid_liq = self.get_book_liquidity(bids, within)
        ask_liq = self.get_book_liquidity(asks, within)

        logger.debug("Bid/ask liq: {}, {}".format(bid_liq, ask_liq))
        logger.debug("Known bids: {}".format(self.known_bids[exchange_id][symbol]))
        logger.debug("Known asks: {}".format(self.known_asks[exchange_id][symbol]))

        return bid_liq, ask_liq

    def get_book_liquidity(self, book: Iterable[Tuple[int, int]], within: float) -> int:
        """
        Helper function for the above.  Checks one side of the known order book.

        Arguments:
            book:
            within:
        """
        liq = 0
        if not book:
            return liq
        best = book[0][0]
        for price, shares in book:
            # Is this price within "within" proportion of the best price?
            if abs(best - price) <= int(round(best * within)):
                logger.debug(
                    "Within {} of {}: {} with {} shares".format(
                        within, best, price, shares
                    )
                )
                liq += shares

        return liq

    def mark_to_market(self, holdings_by_exchange: Dict[int, Dict[str, int]], cash: int,
                       withdrawal_fees: Dict[int, Dict[str, int]]) -> int:
        """
        Marks holdings to market using the best net liquidation value for each asset,
        accounting for withdrawal fees.

        Arguments:
            holdings_by_exchange: The agent's segregated holdings.
            cash: The agent's central cash pool.
            withdrawal_fees: A dictionary mapping symbols to their withdrawal fees.
        """
        total_value = cash

        # Iterate through each exchange where assets are held
        for exchange_id, holdings in holdings_by_exchange.items():
            for symbol, shares in holdings.items():
                if shares == 0:
                    continue

                # Get the withdrawal fee for this specific asset
                fee = withdrawal_fees[exchange_id].get(symbol, withdrawal_fees[exchange_id].get('default', 0))

                best_price = self.get_best_net_price(symbol, exchange_id, fee)

                asset_value = shares * best_price
                total_value += asset_value

                self.logEvent(
                    "MARK_TO_MARKET",
                    f"{shares} {symbol} @ {best_price} == {asset_value}"
                )

        self.logEvent("MARKED_TO_MARKET", total_value)

        return total_value

    def get_best_net_price(self, symbol: str, current_exchange: int, withdrawal_fee: int) -> int:
        """
        Finds the best net liquidation price for a symbol by checking all possible
        markets and subtracting withdrawal fees if a transfer is needed.

        Arguments:
            symbol: The asset to be priced.
            current_exchange: The exchange where the asset is currently held.
            withdrawal_fee: The fee to move this asset from its current exchange.
        """
        best_net_price = 0

        # Check the price on every exchange
        for potential_market_id in self.exchange_ids:
            # Get the price for the symbol on the potential market
            market_price = self.last_trade.get(potential_market_id, {}).get(symbol, 0)

            if market_price == 0:
                continue  # This market doesn't trade the symbol

            # If the asset is already at this market, the fee is zero.
            # Otherwise, we must subtract the withdrawal fee.
            transfer_cost = 0 if potential_market_id == current_exchange else withdrawal_fee

            net_price = market_price - transfer_cost

            # Keep track of the highest possible net price
            if net_price > best_net_price:
                best_net_price = net_price

        return best_net_price

    def get_holdings(self, symbol: str) -> int:
        """
        Gets holdings.  Returns zero for any symbol not held.

        Arguments:
            symbol: The symbol to query.
        """
        total_shares = 0
        for exchange_id, holdings in self.holdings_by_exchange.items():
            total_shares += holdings.get(symbol,0)
        return total_shares

    def get_holdings_by_exchange(self, symbol: str, exchange_id: int) -> int:
        """
        Safely gets the holdings for a symbol on a specific exchange.
        Returns 0 if the exchange or symbol is not found.
        """
        return self.holdings_by_exchange.get(exchange_id, {}).get(symbol, 0)

    def get_known_bid_ask_midpoint(
        self, exchange_id: int, symbol: str
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Get the known best bid, ask, and bid/ask midpoint from cached data. No volume.

        Arguments:
            symbol: The symbol to query.
        """

        bids, asks = self.get_known_bid_ask(exchange_id, symbol, best=False)

        bid = bids[0][0] if bids else None
        ask = asks[0][0] if asks else None

        midpoint = (
            int(round((bid + ask) / 2)) if bid is not None and ask is not None else None
        )

        return bid, ask, midpoint

    def get_average_transaction_price(self) -> float:
        """Calculates the average price paid (weighted by the order size)."""

        return round(
            sum(
                executed_order.quantity * executed_order.fill_price
                for executed_order in self.executed_orders
            )
            / sum(executed_order.quantity for executed_order in self.executed_orders),
            2,
        )

    def fmt_holdings(self, holdings_by_exchange: Mapping[int, Mapping[str, int]]) -> str:
        """
        Prints holdings segregated by exchange and the central cash pool.
        """
        # A list to hold the formatted string for each exchange's holdings.
        exchange_strings = []

        # Loop through each exchange and its dictionary of assets.
        for exchange_id, holdings in sorted(holdings_by_exchange.items()):
            if not holdings:
                continue

            # Format the assets within the exchange, e.g., "BTC: 10, ETH: 5".
            asset_strings = [f"{symbol}: {quantity}" for symbol, quantity in sorted(holdings.items())]

            # Create the complete string for one exchange.
            # e.g., "Exchange exchange_A: { BTC: 10 }"
            exchange_strings.append(f"Exchange {exchange_id}: {{ {', '.join(asset_strings)} }}")

        # Join the strings for all exchanges together.
        holdings_str = ", ".join(exchange_strings)

        # Format the central cash balance.
        cash_str = f"CASH: {self.cash}"

        # Combine everything into the final string.
        # e.g., "{ Exchange exchange_A: { BTC: 10 }, CASH: 50000 }"
        if holdings_str:
            final_str = f"{{ {holdings_str}, {cash_str} }}"
        else:
            # Handle case where there are no asset holdings yet.
            final_str = f"{{ {cash_str} }}"

        return final_str
