from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator, InterArrivalTimeGenerator

# This agent now inherits from the newly refactored CoreBackgroundAgent
from ..agents.core_background_agent import CoreBackgroundAgent

# If init goes remove this
from abides_markets.orders import Order

from .core_gym_agent import CoreGymAgent


class FinancialGymAgent(CoreBackgroundAgent, CoreGymAgent):
    """
    An agent that serves as the primary interface between the ABIDES simulation
    and an external Gym environment (e.g., for Reinforcement Learning).

    This agent inherits all multi-exchange, fee-aware, and segregated-holding
    capabilities from its parent, CoreBackgroundAgent. Its main role is to
    prepare the simulation state for the Gym at each step and to apply actions
    (which must now include an `exchange_id`) received from the Gym.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        subscribe_freq: int = int(1e8),
        subscribe: bool = True,
        subscribe_num_levels: int = 10,
        wakeup_interval_generator: InterArrivalTimeGenerator = ConstantTimeGenerator(
            step_duration=str_to_ns("1min")
        ),
        state_buffer_length: int = 2,
        market_data_buffer_length: int = 5,
        first_interval: Optional[NanosecondTime] = None,
        log_orders: bool = False,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        """
        The __init__ method is now greatly simplified. It passes all parameters
        to the parent CoreBackgroundAgent, which correctly initializes all
        necessary attributes for the multi-exchange environment.
        """
        super().__init__(
            id=id,
            symbol=symbol,
            starting_cash=starting_cash,
            subscribe_freq=subscribe_freq,
            subscribe=subscribe,
            subscribe_num_levels=subscribe_num_levels,
            wakeup_interval_generator=wakeup_interval_generator,
            state_buffer_length=state_buffer_length,
            market_data_buffer_length=market_data_buffer_length,
            first_interval=first_interval,
            log_orders=log_orders,
            name=name,
            type=type,
            random_state=random_state,
        )
        # TODO: No redundant attribute initializations are needed here
        self.symbol: str = symbol
        # Frequency of agent data subscription up in ns-1
        self.subscribe_freq: int = subscribe_freq
        self.subscribe: bool = subscribe
        self.subscribe_num_levels: int = subscribe_num_levels

        self.wakeup_interval_generator: InterArrivalTimeGenerator = (
            wakeup_interval_generator
        )
        self.lookback_period: NanosecondTime = self.wakeup_interval_generator.mean()

        if hasattr(self.wakeup_interval_generator, "random_generator"):
            self.wakeup_interval_generator.random_generator = self.random_state

        self.state_buffer_length: int = state_buffer_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: Optional[NanosecondTime] = first_interval
        # internal variables
        self.has_subscribed: bool = False
        self.episode_executed_orders: List[
            Order
        ] = []  # list of executed orders during full episode

        # list of executed orders between steps - is reset at every step
        self.inter_wakeup_executed_orders: List[Order] = []
        self.parsed_episode_executed_orders: List[Tuple[int, int]] = []  # (price, qty)
        self.parsed_inter_wakeup_executed_orders: List[
            Tuple[int, int]
        ] = []  # (price, qty)
        self.parsed_mkt_data: Dict[str, Any] = {}
        self.parsed_mkt_data_buffer = deque(maxlen=self.market_data_buffer_length)
        self.parsed_volume_data = {}
        self.parsed_volume_data_buffer = deque(maxlen=self.market_data_buffer_length)
        self.raw_state = deque(maxlen=self.state_buffer_length)
        # dictionary to track order status:
        # - keys = order_id
        # - value = dictionary {'active'|'cancelled'|'executed', Order, 'active_qty','executed_qty', 'cancelled_qty }
        self.order_status: Dict[int, Dict[str, Any]] = {}


    def act_on_wakeup(self) -> Dict:
        """
        This method is called by the parent's `wakeup` logic. It prepares the
        state to be sent to the Gym environment.

        The logic is simplified as the parent's `wakeup` method now handles
        scheduling the next wakeup call.
        """
        # 1. Update the agent's internal raw state with the latest market
        #    and internal data. This is handled by a parent method.
        self.update_raw_state()

        # 2. Get a deep copy of the state to return to the Gym.
        #    This state now correctly reflects the multi-exchange environment
        #    (e.g., `holdings_by_exchange`).
        raw_state = deepcopy(self.get_raw_state())

        # 3. Reset the inter-step buffers to prepare for the next step.
        self.new_step_reset()

        # 4. Return the state. The kernel will pass this out to the Gym.
        return raw_state