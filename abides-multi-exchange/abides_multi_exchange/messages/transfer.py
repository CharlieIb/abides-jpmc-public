from dataclasses import dataclass
from abides_core import Message


@dataclass
class CompleteTransferMsg(Message):
    """A custom message to signal the completion of an asset transfer."""

    to_exchange: int
    symbol: str
    size: int
