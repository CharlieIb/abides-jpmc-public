import numpy as np

# Import to register environments
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
        self.max_position = 100  # 0: No position (flat), 1: Long position
        self.stop_loss_pct = 0.05

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

        total_holdings = info.get('total_holdings', 0)
        current_cash = info.get("cash", 0)
        action = 0 # DEFAULT action is HOLD
        holdings_by_exchange = info.get("holdings_by_exchange", {})


        # --- Bollinger Band Mean Reversion Trading Logic ---

        # House keeping, net offset, if both long and short positions
        action = self._net_offsetting_inventory(info)
        if action != 0:
            return action


        # check for exit signals on any existing positions
        exit_signal_found = False
        if total_holdings > 0:  # LONG position
            exchange_with_assets = max(holdings_by_exchange,
                                       key=lambda ex: holdings_by_exchange[ex].get(self.symbol, 0))
            # Stop-Loss
            if global_mid_price <= current_lb * (1 - self.stop_loss_pct):
                action = 2 + (exchange_with_assets * 2)
                exit_signal_found = True
            # Take-Profit
            elif global_mid_price >= current_mb:
                action = 2 + (exchange_with_assets * 2)
                exit_signal_found = True

        elif total_holdings < 0:  # SHORT position
            exchange_with_assets = min(holdings_by_exchange,
                                       key=lambda ex: holdings_by_exchange[ex].get(self.symbol, 0))
            # Stop-Loss
            if global_mid_price >= current_ub * (1 + self.stop_loss_pct):
                action = 1 + (exchange_with_assets * 2)  # Buy to cover
                exit_signal_found = True
            # Take-Profit
            elif global_mid_price <= current_mb:
                action = 1 + (exchange_with_assets * 2)
                exit_signal_found = True

        # If no exit was triggered, look for entry signals to open or ADD to a position
        if not exit_signal_found:
            # Check for BUY signal
            if global_mid_price <= current_lb:
                # Condition: Are we below our max long position?
                if total_holdings < self.max_position:
                    buy_size = min(self.order_fixed_size, self.max_position-total_holdings)
                    cost_to_buy = buy_size * best_global_ask
                    if buy_size > 0 and current_cash is not None and current_cash >= cost_to_buy:
                        action = 1 + (best_buy_exchange * 2)  # BUY on cheapest exchange

            # Check for SELL SHORT signal
            elif global_mid_price >= current_ub:
                # Condition: Are we below our max short position (in absolute terms)?
                if abs(total_holdings) < self.max_position:
                    sell_size = min(self.order_fixed_size, self.max_position - abs(total_holdings))
                    if sell_size > 0:
                        action = 2 + (best_sell_exchange * 2)  # SELL on most expensive exchange

        # If nothing else, consolidate
        if action == 0 and total_holdings:
            action = self._consolidate_inventory(info)

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

    def _consolidate_inventory(self, info: dict) -> int:
        """
        Checks if assets should be moved from a high-fee exchange to a low-fee one.
        Returns a TRANSFER action if needed, otherwise returns 0 (HOLD).
        """
        holdings_by_exchange = info.get("holdings_by_exchange", {})
        withdrawal_fees = info.get("withdrawal_fees", {})

        exchanges_with_assets = [
            ex_id for ex_id, holdings in holdings_by_exchange.items()
            if holdings.get(self.symbol, 0) > 0
        ]

        # Check if assets are already consolidated
        if len(exchanges_with_assets) <= 1:
            return 0  # HOLD

        exchange_ids = list(range(self.num_exchanges))
        # "Home Base" is the cheapest exchange
        cheapest_exchange = min(
            exchange_ids,
            key=lambda ex: withdrawal_fees.get(ex, {}).get(self.symbol, float('inf'))
        )

        most_expensive_source = max(
            exchanges_with_assets,
            key=lambda ex: withdrawal_fees.get(ex, {}).get(self.symbol, -1.0)
        )

        # Transfer from most expensive to cheapest
        if most_expensive_source != cheapest_exchange:
            if most_expensive_source == 0 and cheapest_exchange == 1:
                # print(f"  MR Agent (Housekeeping): Consolidating inventory from expensive Ex 0 to cheap Ex 1.")
                return 6
            elif most_expensive_source == 1 and cheapest_exchange == 0:
                # print(f"  MR Agent (Housekeeping): Consolidating inventory from expensive Ex 1 to cheap Ex 0.")
                return 5

        return 0


    def _net_offsetting_inventory(self, info: dict) -> int:
        """
        Checks for and resolves offsetting long/short positions across exchanges.
        This is a high-priority risk management action.
        Returns a TRANSFER action if needed, otherwise returns 0 (HOLD).
        """

        holdings_by_exchange = info.get("holdings_by_exchange", {})

        long_exchanges = {ex: h.get(self.symbol, 0) for ex, h in holdings_by_exchange.items() if h.get(self.symbol, 0) > 0}
        short_exchanges = {ex: h.get(self.symbol, 0) for ex, h in holdings_by_exchange.items() if h.get(self.symbol, 0) < 0}

        # If we have both a long and a short position somewhere, we need to net them.
        if long_exchanges and short_exchanges:
            # Find the largest long and largest short positions to resolve first.
            from_exchange = max(long_exchanges, key=long_exchanges.get)
            to_exchange = min(short_exchanges, key=short_exchanges.get)

            # The amount to transfer is the smaller of the two positions to close one leg out.
            transfer_size = min(long_exchanges[from_exchange], abs(short_exchanges[to_exchange]))

            if transfer_size > 0:
                # This logic assumes a 2-exchange setup with actions:
                # 5: TRANSFER_FROM_0_TO_1
                # 6: TRANSFER_FROM_1_TO_0
                if from_exchange == 0 and to_exchange == 1:
                    print(
                        f"  MR Agent (Netting): Transferring from Ex 0 to Ex 1 to flatten position.")
                    return 6
                elif from_exchange == 1 and to_exchange == 0:
                    print(
                        f"  MR Agent (Netting): Transferring shares from Ex 1 to Ex 0 to flatten position.")
                    return 5

        return 0

