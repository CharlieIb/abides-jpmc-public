import gym
from tqdm import tqdm
import numpy as np

# Import to register environments
import abides_gym
import gym_crypto_markets


# Helper function to calculate Bollinger Bands
def calculate_bollinger_bands(prices, window=300, num_std_dev=2):
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

    def __init__(self, observation_space, action_space, window=300, num_std_dev=2, num_exchanges=2):
        self.observation_space = observation_space
        self.action_space = action_space

        self.window = window  # Period for SMA and Std Dev calculation
        self.num_std_dev = num_std_dev  # Number of standard deviations for bands

        self.num_exchanges = num_exchanges
        self.symbol = "ABM"

        self.price_history = []  # To store recent absolute prices for band calculation
        self.position = 0  # 0: No position (flat), 1: Long position

        # Store the order size the environment will use when we send BUY/SELL actions
        self.order_fixed_size = 10  # Default, will be updated from env.unwrapped.order_fixed_size

        print(
            f"MeanReversionAgent initialized with obs space: {observation_space.shape} and action space: {action_space.n if hasattr(action_space, 'n') else action_space.shape}")
        print(f"Bollinger Band Parameters: Window={self.window}, Std Devs={self.num_std_dev}")


    def choose_action(self, state, info):
        """
        Chooses an action based on Bollinger Band mean reversion rules.

        State structure (for 2 exchanges, more exchanges means more features and different structure):
        GLOBAL features
        state[0]: total_holdings (e.g., [[80.]]) -> scalar value at state[0][0]
        Exchange 1 features
        state[1]: imbalance (e.g., [0.18690784]) -> scalar value at state[1][0]
        state[2]: spread (e.g., [1.]) -> scalar value at state[2][0]
        state[3]: directionFeature (e.g., [-0.5]) -> scalar value at state[3][0]
        Exchange 2 features
        state[4]: imbalance (e.g., [0.18690784]) -> scalar value at state[1][0]
        state[5]: spread (e.g., [1.]) -> scalar value at state[2][0]
        state[6]: directionFeature (e.g., [-0.5]) -> scalar value at state[3][0]
        state[7-8]: Rk_feature_t (price changes, e.g., [[0.], [2.], [-1.]]) -> array at state[4]
                  (Note: Rk_feature_t not directly used in this specific BB strategy, but part of state)

        Info dictionary (requires debug_mode=True in env):
        info["cash"]: Current cash balance
        info["last_transaction"]: Current absolute price of the asset
        """
        # Logic to find the best global bid and ask
        best_global_bid, best_global_ask = -1.0, float('inf')
        best_sell_exchange, best_buy_exchange = None, None

        # Extract necessary information
        market_data_all = info.get("market_data", {})
        for ex_id in range(self.num_exchanges):
            market_data = market_data_all.get(ex_id, {})

            best_bid = market_data.get("best_bid")
            best_ask = market_data.get("best_ask")

            if best_bid is not None and best_bid > best_global_bid:
                best_global_bid = best_bid
                best_sell_exchange = ex_id

            if best_ask is not None and best_ask < best_global_ask:
                best_global_ask = best_ask
                best_buy_exchange = ex_id

        # No valid price on any exchange, HOLD.
        if best_global_bid == -1.0 or best_global_ask == float('inf'):
            return 0  # HOLD

        # Calculate a single, robust mid-price.
        global_mid_price = (best_global_bid + best_global_ask) / 2
        self.price_history.append(global_mid_price)

        if len(self.price_history) < self.window:
            return 0

        if len(self.price_history) > self.window * 2:
            self.price_history.pop(0)

        middle_band, upper_band, lower_band = calculate_bollinger_bands(
            self.price_history, self.window, self.num_std_dev
        )

        # --- Latest band values. ---
        # (These should not be NaN if len(price_history) >= window)
        current_mb = middle_band[-1]
        current_ub = upper_band[-1]
        current_lb = lower_band[-1]

        current_cash = info.get("cash")
        action = 0 # DEFAULT action is HOLD

        # --- Bollinger Band Mean Reversion Trading Logic ---
        if self.position == 0:  # Agent is currently flat (no position)
            # Buy Signal: Price crosses below or touches the Lower Band (oversold)
            if global_mid_price <= current_lb:
                # Check if we have enough cash to buy the fixed order size (with a small buffer for fees)
                cost_to_buy = self.order_fixed_size * best_global_ask
                if (current_cash is not None and
                        current_cash >= cost_to_buy and
                        best_buy_exchange is not None):

                    action = (1 + (best_buy_exchange * 2)) # if EX 0 action = 1 if  EX_1 action = 3 this is BUY on respective exchanges
                    self.position = 1  # agent is now long
                    print(f"  MR Agent: BUY signal! Price {best_global_ask:.2f} <= LB {current_lb:.2f}. Cash: {current_cash:.2f}. Cost: {cost_to_buy:.2f}")
                else:
                    print(f"  MR Agent: BUY signal, but insufficient cash ({current_cash:.2f}) for cost {cost_to_buy:.2f}.")


        elif self.position == 1:  # Agent is long, looking to SELL
            if global_mid_price >= current_mb:
                # Agent's holding structure is determined.
                # For simplicity, find the exchange with the most shares.
                holdings_by_exchange = info.get("holdings_by_exchange", {})
                if not holdings_by_exchange:
                    return 0  # No holdings to sell
                exchange_with_assets = max(holdings_by_exchange,
                                           key=lambda ex: holdings_by_exchange[ex].get(self.symbol, 0))

                # --- Fee structure retrieval ---
                withdrawal_fees = info.get("withdrawal_fees", {})
                fee_structure = withdrawal_fees.get(exchange_with_assets, {})
                fee = fee_structure.get(self.symbol, fee_structure.get('default', 0))

                # --- Transfer cost calculation ---
                transfer_cost = 0
                if best_sell_exchange != exchange_with_assets:
                    transfer_cost = fee

                net_sell_price = best_global_bid - transfer_cost

                # Sell if still attractive
                if net_sell_price >= current_mb and best_sell_exchange is not None:
                    action = 2 + (best_sell_exchange * 2)  # SELL on best exchange
                    self.position = 0

                    print(f"  MR Agent: SELL signal (close long)! Price {best_global_bid:.2f} >= MB {current_mb:.2f}. Holdings: {exchange_with_assets:}.")
                else:
                    print(f"  MR Agent: SELL signal, but no holdings to sell.")

        return action

    def update_policy(self, old_state, action, reward, new_state, done):
        """
        No learning for rule-based agent.
        """
        pass

    def reset_agent_state(self):
        """Resets the agent's internal state for a new episode."""
        self.price_history = []
        self.position = 0



