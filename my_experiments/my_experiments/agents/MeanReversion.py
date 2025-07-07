import gym
from tqdm import tqdm
import numpy as np

# Import to register environments
import abides_gym


# Helper function to calculate Bollinger Bands
def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    """
    Calculates Bollinger Bands for a given series of prices.
    Returns (middle_band, upper_band, lower_band) as numpy arrays.
    Handles NaN for initial periods where data is insufficient.
    """
    prices_series = np.array(prices, dtype=np.float32)  # Ensure float32

    if len(prices_series) < window:
        # Not enough data for calculation, return arrays of NaNs
        nan_array = np.full(len(prices_series), np.nan)
        return nan_array, nan_array, nan_array

    # Calculate Simple Moving Average (Middle Band)
    # Use convolution for a rolling mean
    weights = np.ones(window) / window
    middle_band_raw = np.convolve(prices_series, weights, mode='valid')

    # Pad the middle_band with NaNs at the beginning to match original price_history length
    padded_middle_band = np.pad(middle_band_raw, (window - 1, 0), 'constant', constant_values=np.nan)

    # Calculate Standard Deviation for the bands
    std_dev_raw = np.array([np.std(prices_series[i - window + 1:i + 1]) for i in range(window - 1, len(prices_series))],
                           dtype=np.float32)
    padded_std_dev = np.pad(std_dev_raw, (window - 1, 0), 'constant', constant_values=np.nan)

    upper_band = padded_middle_band + (num_std_dev * padded_std_dev)
    lower_band = padded_middle_band - (num_std_dev * padded_std_dev)

    return padded_middle_band, upper_band, lower_band


# --- Define your Mean Reversion Agent Class ---
class MeanReversionAgent:
    """
    A rule-based trading agent using a Bollinger Band Mean Reversion strategy.
    This agent buys when prices are oversold (below lower band)
    and sells when they revert to the mean (above middle band).
    It is a long-only strategy and serves as a baseline.
    Requires 'cash' and 'last_transaction' in the info dictionary, which
    becomes available after the first env.step().
    """

    def __init__(self, observation_space, action_space, window=20, num_std_dev=2):
        self.observation_space = observation_space
        self.action_space = action_space

        self.window = window  # Period for SMA and Std Dev calculation
        self.num_std_dev = num_std_dev  # Number of standard deviations for bands

        self.price_history = []  # To store recent absolute prices for band calculation
        self.position = 0  # 0: No position (flat), 1: Long position
        self.initial_cash = 0  # Will be set from env.unwrapped.starting_cash on reset

        # Store the order size the environment will use when we send BUY/SELL actions
        self.order_fixed_size = 10  # Default, will be updated from env.unwrapped.order_fixed_size

        print(
            f"MeanReversionAgent initialized with obs space: {observation_space.shape} and action space: {action_space.n if hasattr(action_space, 'n') else action_space.shape}")
        print(f"Bollinger Band Parameters: Window={self.window}, Std Devs={self.num_std_dev}")
        print(
            "Note: This agent assumes the state structure is (holdings, imbalance, spread, directionFeature, Rk_feature_t).")
        print(
            "It relies on 'cash' and 'last_transaction' being available in the info dictionary (requires debug_mode=True).")
        print("Agent will HOLD until enough price history is accumulated for Bollinger Bands.")

    def choose_action(self, state, info):
        """
        Chooses an action based on Bollinger Band mean reversion rules.

        State structure:
        state[0]: holdings (e.g., [[80.]]) -> scalar value at state[0][0]
        state[1]: imbalance (e.g., [0.18690784]) -> scalar value at state[1][0]
        state[2]: spread (e.g., [1.]) -> scalar value at state[2][0]
        state[3]: directionFeature (e.g., [-0.5]) -> scalar value at state[3][0]
        state[4]: Rk_feature_t (price changes, e.g., [[0.], [2.], [-1.]]) -> array at state[4]
                  (Note: Rk_feature_t not directly used in this specific BB strategy, but part of state)

        Info dictionary (requires debug_mode=True in env):
        info["cash"]: Current cash balance
        info["last_transaction"]: Current absolute price of the asset
        """
        # Extract necessary information
        current_holdings = state[0][0]  # Holdings are in the state tuple
        current_cash = info.get("cash")  # Cash from info dict
        current_price = info.get("last_transaction")  # Last transaction price from info dict

        action = 1  # Default action is HOLD (action ID 1 for HOLD in ABIDES)

        # CRITICAL: Handle cases where info might not yet contain price/cash (e.g., first step after reset)
        if current_cash is None or current_price is None:
            # print("  MR Agent: Cash or Last Transaction Price info missing (likely first step). HOLD.")
            return action

        # Add current price to history only if it's not None
        self.price_history.append(current_price)

        # Ensure we have enough data for Bollinger Band calculation
        if len(self.price_history) < self.window:
            # Not enough data for full window calculation, just hold
            # print(f"  MR Agent: Accumulating price history ({len(self.price_history)}/{self.window} prices). HOLD.")
            return action

        # Keep the price history size manageable (e.g., last 2*window prices)
        # This prevents the history list from growing infinitely, important for long simulations.
        if len(self.price_history) > self.window * 2:
            self.price_history.pop(0)

        # Calculate Bollinger Bands based on current price history
        middle_band, upper_band, lower_band = calculate_bollinger_bands(
            self.price_history, self.window, self.num_std_dev
        )

        # Get the latest band values. These should not be NaN if len(price_history) >= window
        current_mb = middle_band[-1]
        current_ub = upper_band[-1]
        current_lb = lower_band[-1]

        # --- Bollinger Band Mean Reversion Trading Logic ---

        if self.position == 0:  # Agent is currently flat (no position)
            # Buy Signal: Price crosses below or touches the Lower Band (oversold)
            if current_price <= current_lb:
                # Check if we have enough cash to buy the fixed order size (with a small buffer for fees)
                cost_to_buy = self.order_fixed_size * current_price
                if current_cash >= cost_to_buy:
                    action = 0  # Action: BUY (action ID 0 for BUY in ABIDES)
                    self.position = 1  # Mark that we are now long
                    print(f"  MR Agent: BUY signal! Price {current_price:.2f} <= LB {current_lb:.2f}. Cash: {current_cash:.2f}. Cost: {cost_to_buy:.2f}")
                else:
                    print(f"  MR Agent: BUY signal, but insufficient cash ({current_cash:.2f}) for cost {cost_to_buy:.2f}.")


        elif self.position == 1:  # Agent is currently long (holding shares)
            # Sell Signal (Close Long): Price crosses above or touches the Middle Band (reversion to mean)
            if current_price >= current_mb:
                if current_holdings > 0:  # Ensure we actually have shares to sell
                    action = 2  # Action: SELL (close entire position) (action ID 2 for SELL in ABIDES)
                    self.position = 0  # Mark that we are now flat
                    print(f"  MR Agent: SELL signal (close long)! Price {current_price:.2f} >= MB {current_mb:.2f}. Holdings: {current_holdings:.2f}.")
                else:
                    print(f"  MR Agent: SELL signal, but no holdings to sell.")

        return action

    def update_policy(self, old_state, action, reward, new_state, done):
        """
        For a rule-based agent, this method typically does not perform learning.
        It can be used to update internal state variables or log information.
        """
        # No policy learning occurs for a rule-based agent.
        pass

    def reset_agent_state(self):
        """Resets the agent's internal state for a new episode."""
        self.price_history = []
        self.position = 0
        self.initial_cash = 10000000000



