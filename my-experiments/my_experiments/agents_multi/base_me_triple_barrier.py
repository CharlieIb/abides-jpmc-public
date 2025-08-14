import numpy as np
from collections import deque

class MultiExchangeTripleBarrierAgent:
    """
    Applies the Triple-Barrier Method across 2 exchanges, seeking the best
    entry and exit prices using a hardcoded action space (0-6).
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 profit_take_percent: float = 0.5,
                 stop_loss_percent: float = 0.3,
                 time_limit_steps: int = 20,
                 max_long_positions: int = 10,
                 max_short_positions: int = 5):

        # Limits on positions
        self.max_long_positions = max_long_positions
        self.max_short_positions = max_short_positions

        # Triple barrier variables
        self.profit_take_pct = float(profit_take_percent) / 100.0
        self.stop_loss_pct = float(stop_loss_percent) / 100.0
        self.time_limit_steps = int(time_limit_steps)
        self.num_exchanges = 2

        self.long_positions = deque()
        self.short_positions = deque()
        self.current_step = 0

        # Diagnostic counters
        self.profit_exits = 0
        self.loss_exits = 0
        self.time_exits = 0

        print(f"MultiPositionAgent initialized with limits: {max_long_positions} Long, {max_short_positions} Short.")

    def _get_all_exchange_prices(self, observation: np.ndarray) -> list:
        prices = []
        global_vwap = observation[0][0]
        prices.append(global_vwap + observation[7][0])
        prices.append(global_vwap + observation[10][0])
        return prices

    def _get_entry_signal(self, observation: np.ndarray) -> str:
        most_recent_return = observation[-1][0]
        if most_recent_return > 0.0001:
            return 'BUY'
        elif most_recent_return < -0.0001:
            return 'SELL'
        return 'HOLD'

    def _create_new_trade(self, entry_price, direction):
        # Helper to create a trade dictionary
        trade = {"direction": direction, "entry_price": entry_price, "entry_step": self.current_step}
        if direction == "LONG":
            trade["top_barrier"] = entry_price * (1 + self.profit_take_pct)
            trade["bottom_barrier"] = entry_price * (1 - self.stop_loss_pct)
        else: # SHORT
            trade["top_barrier"] = entry_price * (1 - self.profit_take_pct)
            trade["bottom_barrier"] = entry_price * (1 + self.stop_loss_pct)
        return trade

    def choose_action(self, observation: np.ndarray, info=None) -> int:
        self.current_step += 1
        all_prices = self._get_all_exchange_prices(observation)
        best_ask = min(all_prices) # Price to buy at
        best_bid = max(all_prices) # Price to sell at

        # --- 1. CHECK FOR EXITS (HIGHEST PRIORITY) ---
        # (This section is unchanged - the agent should always be able to exit)
        for i, trade in reversed(list(enumerate(self.long_positions))):
            if best_bid >= trade["top_barrier"]:
                self.profit_exits += 1
                del self.long_positions[i]
                return 2 + (2 * np.argmax(all_prices))
            if best_bid <= trade["bottom_barrier"]:
                self.loss_exits += 1
                del self.long_positions[i]
                return 2 + (2 * np.argmax(all_prices))
            if self.current_step > trade["entry_step"] + self.time_limit_steps:
                self.time_exits += 1
                del self.long_positions[i]
                return 2 + (2 * np.argmax(all_prices))

        for i, trade in reversed(list(enumerate(self.short_positions))):
            if best_ask <= trade["top_barrier"]:
                self.profit_exits += 1
                del self.short_positions[i]
                return 1 + (2 * np.argmin(all_prices))
            if best_ask >= trade["bottom_barrier"]:
                self.loss_exits += 1
                del self.short_positions[i]
                return 1 + (2 * np.argmin(all_prices))
            if self.current_step > trade["entry_step"] + self.time_limit_steps:
                self.time_exits += 1
                del self.short_positions[i]
                return 1 + (2 * np.argmin(all_prices))

        # Check for NEW ENTRY positions
        signal = self._get_entry_signal(observation)

        cash = observation[4][0]  # Get current cash from the observation

        order_size = 100  # currently because of fixed order size 100
        estimated_cost = order_size * best_ask


        # Check for a new LONG position
        if signal == 'BUY' and len(self.long_positions) < self.max_long_positions and cash > estimated_cost:
            buy_exchange_id = np.argmin(all_prices)
            new_trade = self._create_new_trade(best_ask, "LONG")
            self.long_positions.append(new_trade)
            print(f"Opening new LONG position ({len(self.long_positions)}/{self.max_long_positions}).")
            return 1 + (2 * buy_exchange_id) # Return BUY action

        # Check for a new SHORT position
        if signal == 'SELL' and len(self.short_positions) < self.max_short_positions:
            sell_exchange_id = np.argmax(all_prices)
            new_trade = self._create_new_trade(best_bid, "SHORT")
            self.short_positions.append(new_trade)
            print(f"Opening new SHORT position ({len(self.short_positions)}/{self.max_short_positions}).")
            return 2 + (2 * sell_exchange_id) # Return SELL action

        # HOLD
        return 0

    def get_episode_diagnostics(self):
        total_trades = self.profit_exits + self.loss_exits + self.time_exits
        win_rate = self.profit_exits / total_trades if total_trades > 0 else 0
        return {
            "total_trades_closed": total_trades,
            "win_rate": win_rate,
            "profit_exits": self.profit_exits,
            "loss_exits": self.loss_exits,
            "time_exits": self.time_exits
        }

    def reset_agent_state(self):
        self.long_positions.clear()
        self.short_positions.clear()
        self.current_step = 0
        self.profit_exits = 0
        self.loss_exits = 0
        self.time_exits = 0

    def update_policy(self,state, action_output, reward, new_state, done):
        pass

    def save_weights(self, filepath):
        pass