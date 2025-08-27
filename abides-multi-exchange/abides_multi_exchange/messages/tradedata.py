from dataclasses import dataclass
from typing import List, Dict, Any

from abides_core import NanosecondTime

from abides_markets.messages.marketdata import MarketDataFreqBasedSubReqMsg, MarketDataMsg
from abides_markets.order_book import Side


@dataclass
class TradeDataSubReqMsg(MarketDataFreqBasedSubReqMsg):
    """
    This message requests the creation or cancellation of a subscription to
    individual trade data from an ''ExchangeAgent''.

    It follows a frequency-based model, sending all new trades that hav
    occurred since the last update.
    """
    # Inherits:
    # symbol =
    # cancel =
    # freq =
    # No new fields are needed
    pass

@dataclass
class TradeDataMsg(MarketDataMsg):
    """
    This message returns a list of individual trades as part of a data subscription.

    Each trade is represented as a Dict of (time, price, quantity, side)
    The 'side represents the side of the resting order that was filled --
    NOT the aggressor
    """
    trades: List[Dict[str, Any]]
    exchange_id: int

