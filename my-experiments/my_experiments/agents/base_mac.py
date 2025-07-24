import numpy as np


def calculate_moving_average(prices, window):
    """
    Calculates the simple moving average for a series of prices.
    returns a numpy array of the same length as prices, with NaNs at the beginning
    """
    if len(prices) < window:
        return np.full(len(prices), np.nan)

    weights = np.ones(window) / window
    sma_raw = np.convolve(np.array(prices, dtype=np.float32), weights, mode='valid')

    sma_padded = np.pad(sma_raw, (window -1, 0), 'constant', constant_values=np.nan)
    return sma_padded

class MovingAverageCrossoverAgent:
    """
    A rule-based trading agent using a Moving Average Crossover strategy.
    This agent buys when a short-term MA crosses above a long-term MA (a "golden cross")
    and sell when the short-term MA crosses below the long-term MA (a "death cross").
    """
    def __init__(self, observation_space, action_space, short_window=9, long_window=21):
        self.observation_space = observation_space
        self.action_space = action_space
        self.short_window = short_window
        self.long_window = long_window

        self.price_history = []
        self.position = 0 # 0: No position, 1: Long position
        self.initial_cash = 0
        self.order_fixed_size = 10

        print(f"MovingAverageCrossoverAgent initialised.")
        print(f"MA Parameters: Short Window={self.short_window}, Long Window={self.long_window}")
        print(f"Agent will HOLD until enough price history is accumulated for the long MA.")

    def choose_action(self, state, info):
        """
        Choose an action based on the MACO rules using the state and the info
        :param state: the observation state from the gym set up
        :param info: the optional info dictionary from gym
        :return:
        """
        current_holdings = state[0][0]
        current_cash = info.get("cash")
        current_price = info.get("last_transaction")

        action = 1 # Default action is HOLD

        if current_cash is None or current_price is None:
            return action

        self.price_history.append(current_price)

        if len(self.price_history) < self.long_window:
            return action

        if len(self.price_history) > 2 * self.long_window:
            self.price_history.pop(0)

        # Calculate the moving averages
        short_ma = calculate_moving_average(self.price_history, self.short_window)
        long_ma = calculate_moving_average(self.price_history, self.long_window)

        last_short_ma = short_ma[-1]
        prev_short_ma = short_ma[-2]

        last_long_ma = long_ma[-1]
        prev_long_ma = long_ma[-2]


        # BUY signal ("Golden Cross"): Short-term MA crosses below long-term MA
        if prev_short_ma <= prev_long_ma and last_short_ma > last_long_ma:
            if self.position == 0: # Only buy if we are currently flat
                cost_to_buy = self.order_fixed_size * current_price
                if current_cash >= cost_to_buy:
                    action = 0
                    self.position = 1
                    print(f"MA Agent: BUY signal! Short MA ({last_short_ma:.2f}) crossed above Long MA ({last_long_ma:.2f}")
                else:
                    print(f" MA Agent: BUY signal, but insufficient cash")


        # SELL signal ("Death Cross"): Short-term MA crosses below long-term MA
        if prev_short_ma >= prev_long_ma and last_short_ma < last_long_ma:
            if self.position == 1:
                if current_holdings > 0:
                    action = 2
                    self.position = 0
                    print(f"  MA Agent: SELL signal! Short MA ({last_short_ma:.2f}) crossed below Long MA ({last_long_ma:.2f}).")
                else:
                    print(f"  MA Agent: SELL signal, but no holdings to sell.")

        return action

    def update_policy(self, old_state, action, reward, new_state, done):
        """ This agent is rule-based; no learning occurs."""
        pass

    def reset_agent_state(self):
        self.price_history = []
        self.position = 0