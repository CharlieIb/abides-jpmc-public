import numpy as np
from collections import deque

class SingleExchangeTripleBarrierAgent:
    """
    Manages a portfolio of long and short trades on a single exchange.
    Action Space: 0=HOLD, 1=BUY, 2=SELL
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 profit_take_percent: float = 0.5,
                 stop_loss_percent: float = 0.3,
                 time_limit_steps: int = 20,
                 max_long_positions: int = 10,
                 max_short_positions: int = 5):

        # Position limit parameters
        self.max_long_positions = max_long_positions
        self.max_short_positions = max_short_positions

        # Strategy parameters
        self.profit_take_pct = float(profit_take_percent) / 100.0
        self.stop_loss_pct = float(stop_loss_percent) / 100.0
        self.time_limit_steps = int(time_limit_steps)

        # Data structures for managing trades
        self.long_positions = deque()
        self.short_positions = deque()
        self.current_step = 0
        print(f"SingleExchangeMultiPositionAgent initialized with limits: {max_long_positions} Long, {max_short_positions} Short.")

        # for final episode summary statistics
        self.profit_exits = 0
        self.loss_exits = 0
        self.time_exits = 0

    def _get_current_price(self, observation: np.ndarray) -> float:
        """
        Gets the current price for the single exchange (exchange 0).
        """
        global_vwap = observation[0][0]
        # Price deviation for exchange 0 is at index 7
        price_deviation = observation[7][0]
        return global_vwap + price_deviation

    def _get_entry_signal(self, observation: np.ndarray) -> str:
        """
        Gets a trading signal based on VWAP momentum.
        """
        most_recent_return = observation[-1][0]
        if most_recent_return > 0.0001:
            return 'BUY'
        elif most_recent_return < -0.0001:
            return 'SELL'
        return 'HOLD'

    def _create_new_trade(self, entry_price, direction):
        """
        Helper to create a trade dictionary with its barrier levels.
        """
        trade = {"direction": direction, "entry_price": entry_price, "entry_step": self.current_step}
        if direction == "LONG":
            trade["top_barrier"] = entry_price * (1 + self.profit_take_pct)
            trade["bottom_barrier"] = entry_price * (1 - self.stop_loss_pct)
        else:  # SHORT
            trade["top_barrier"] = entry_price * (1 - self.profit_take_pct)
            trade["bottom_barrier"] = entry_price * (1 + self.stop_loss_pct)
        return trade

    def choose_action(self, observation: np.ndarray, info=None) -> int:
        self.current_step += 1
        current_price = self._get_current_price(observation)

        # --- 1. CHECK FOR EXITS (HIGHEST PRIORITY) ---
        # Check to close a long position
        for i, trade in reversed(list(enumerate(self.long_positions))):
            if current_price >= trade["top_barrier"]:
                self.profit_exits += 1  # Increment win counter
                del self.long_positions[i]
                return 2
                # Check for loss exit
            if current_price <= trade["bottom_barrier"]:
                self.loss_exits += 1  # Increment loss counter
                del self.long_positions[i]
                return 2
                # Check for time limit exit
            if self.current_step > trade["entry_step"] + self.time_limit_steps:
                self.time_exits += 1  # Increment time-out counter
                del self.long_positions[i]
                return 2

        # Check to close a short position
        for i, trade in reversed(list(enumerate(self.short_positions))):
            if current_price <= trade["top_barrier"]:
                self.profit_exits += 1
                del self.short_positions[i]
                return 1
                # Check for loss exit
            if current_price >= trade["bottom_barrier"]:
                self.loss_exits += 1
                del self.short_positions[i]
                return 1
                # Check for time limit exit
            if self.current_step > trade["entry_step"] + self.time_limit_steps:
                self.time_exits += 1
                del self.short_positions[i]
                return 1

        # If not exiting check for NEW ENTRY
        signal = self._get_entry_signal(observation)

        # Check for a new LONG position
        if signal == 'BUY' and len(self.long_positions) < self.max_long_positions:
            new_trade = self._create_new_trade(current_price, "LONG")
            self.long_positions.append(new_trade)
            # print(f"Opening new LONG position ({len(self.long_positions)}/{self.max_long_positions}).")
            return 1  # BUY action to open long

        # Check for a new SHORT position
        if signal == 'SELL' and len(self.short_positions) < self.max_short_positions:
            new_trade = self._create_new_trade(current_price, "SHORT")
            self.short_positions.append(new_trade)
            # print(f"Opening new SHORT position ({len(self.short_positions)}/{self.max_short_positions}).")
            return 2  # SELL action to open short

            # HOLD
        return 0

    def get_episode_diagnostics(self):
        """NEW: Returns a dictionary of the episode's performance."""
        total_trades = self.profit_exits + self.loss_exits + self.time_exits
        win_rate = self.profit_exits / total_trades if total_trades > 0 else 0

        return {
            "total_trades_closed": total_trades,
            "win_rate": win_rate,
            "profit_exits": self.profit_exits,
            "loss_exits": self.loss_exits,
            "time_exits": self.time_exits
        }

    def update_policy(self, state, action_output, reward, new_state, done):
        """This is a rule-based agent, so no policy update is needed."""
        pass

    def reset_agent_state(self):
        """Resets the agent's positions for a new episode."""
        self.long_positions.clear()
        self.short_positions.clear()
        self.current_step = 0
        self.profit_exits = 0
        self.loss_exits = 0
        self.time_exits = 0