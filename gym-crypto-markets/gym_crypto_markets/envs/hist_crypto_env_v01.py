import gym
import numpy as np
import pandas as pd
from collections import deque


class HistoricalTradingEnv_v01(gym.Env):
    """
    A Gym environment for training and testing trading agents on historical data.
    It reads a CSV of trade data and simulates the agent's interaction with the market.
    """

    def __init__(self, data_path: str, abides_env_config: dict, starting_cash: int = 10_000_000_000):
        super(HistoricalTradingEnv_v01, self).__init__()

        # --- Environment Parameters ---
        self.starting_cash = starting_cash
        self.data_path = data_path

        # --- Mirror key parameters from the ABIDES config ---
        # This makes the historical env compatible with your runner script
        self.num_exchanges = abides_env_config.get('num_exchange_agents', 2)
        self.state_history_length = abides_env_config.get('state_history_length', 5)

        # Based on your ABIDES env, the state has 17 features
        self.num_state_features = 7 + (self.num_exchanges * 3) + (self.state_history_length - 1)
        # Action space is 7 for 2 exchanges (HOLD, BUY/SELL E0, BUY/SELL E1, TFRs)
        self.num_actions = (self.num_exchanges) ** 2 + self.num_exchanges + 1

        # --- Data Loading ---
        self._load_data()

        # --- State and Action Spaces (must match ABIDES environment) ---
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_state_features, 1), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # --- Internal State Tracking ---
        self.current_step = 0
        self.cash = self.starting_cash
        self.holdings = 0
        self.previous_portfolio_value = self.starting_cash
        # Buffer to store historical VWAPs for return calculation
        self.vwap_history = deque(maxlen=self.state_history_length)

    def _load_data(self):
        """Loads and preprocesses the historical trade data."""
        print(f"Loading historical data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        # Ensure you have 'price' and 'quantity' columns.
        # You might need to preprocess timestamps if you use them.
        print(f"Loaded {len(self.df)} data points.")

    def reset(self):
        """Resets the environment for a new episode."""
        self.current_step = 0
        self.cash = self.starting_cash
        self.holdings = 0
        self.previous_portfolio_value = self.starting_cash

        # Return the initial state
        return self._calculate_state()

    def step(self, action: int):
        """Executes one step in the environment, handling the full ABIDES action space."""
        trade_size = 100  # Assume fixed trade size for simplicity
        current_price = self.df.loc[self.current_step, 'price']

        # Map the full action space to historical execution
        # 0: HOLD, 1: BUY E0, 2: SELL E0, 3: BUY E1, 4: SELL E1, 5: TFR, 6: TFR
        if action == 1:  # BUY on Exchange 0
            cost = current_price * trade_size
            if self.cash >= cost:
                self.cash -= cost
                self.holdings += trade_size
        elif action == 2:  # SELL on Exchange 0
            # Basic check to prevent selling what you don't have
            if self.holdings >= trade_size:
                self.cash += current_price * trade_size
                self.holdings -= trade_size
        # All other actions (0, 3, 4, 5, 6) are treated as HOLD
        # because this environment only simulates one market (Exchange 0).

        # --- Advance the market ---
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # --- Calculate Reward ---
        # Use next step's price for marking-to-market
        next_price = self.df.loc[self.current_step, 'price'] if not done else current_price
        current_portfolio_value = self.cash + (self.holdings * next_price)
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value

        # --- Calculate New State ---
        next_state = self._calculate_state()

        # --- Info dictionary ---
        info = {
            'portfolio_value': current_portfolio_value,
            'cash': self.cash,
            'total_holdings': self.holdings
        }

        return next_state, reward, done, info

    def _calculate_state(self) -> np.ndarray:
        """
        Calculates the state vector to be identical to the ABIDES environment's structure.
        """
        # --- Define the observation window (e.g., last 10 minutes) ---
        # This window is used to calculate features like volatility.
        window_size = 10
        start_index = max(0, self.current_step - window_size)
        window = self.df.iloc[start_index: self.current_step + 1]

        # --- Pre-calculate base values from the current time step's data ---
        if not window.empty:
            # Current minute's data
            current_data = self.df.iloc[self.current_step]
            # Assumes your CSV has 'vwap', 'volume', 'buy_volume' columns from pre-aggregation
            current_vwap = current_data.get('vwap', 0)
            current_total_volume = current_data.get('volume', 0)
            current_buy_volume = current_data.get('buy_volume', 0)

            # Window data
            window_vwaps = self.df['vwap'].iloc[start_index: self.current_step + 1]

            # Calculate volatility from the log returns of VWAPs in the window
            if len(window_vwaps) > 1:
                log_returns = np.log(window_vwaps.iloc[1:].values / window_vwaps.iloc[:-1].values)
                volatility = np.std(log_returns)
            else:
                volatility = 0
        else:
            current_vwap, current_total_volume, current_buy_volume, volatility = 0, 0, 0, 0

        # Update VWAP history for temporal features
        if current_vwap > 0:
            self.vwap_history.append(current_vwap)

        # --- Feature Calculation (in the correct order) ---

        # 1. Global VWAP
        global_vwap = current_vwap
        # 2. Total Market Volume
        total_market_volume = current_total_volume
        # 3. Global Trade Imbalance (TVI)
        global_tvi = current_buy_volume / total_market_volume if total_market_volume > 0 else 0.5
        # 4. Overall Volatility
        global_volatility = volatility
        # 5. Cash
        cash = self.cash
        # 6. Total Holdings
        total_holdings = self.holdings
        # 7. PnL
        current_price = self.df.loc[self.current_step, 'price']
        pnl = (self.cash + self.holdings * current_price) - self.starting_cash

        global_and_agent_features = [
            global_vwap, total_market_volume, global_tvi, global_volatility,
            cash, total_holdings, pnl
        ]

        # Per-Exchange Features (3 features per exchange)
        exchange_features = []
        for ex_id in range(self.num_exchanges):
            if ex_id == 0:  # This is our simulated historical market
                price_dev = 0  # No deviation, as it's the global VWAP
                vol_share = 1.0  # It has 100% of the volume
                tvi = global_tvi  # Same as global TVI
                exchange_features.extend([price_dev, vol_share, tvi])
            else:  # For all other exchanges, append placeholder zeros
                exchange_features.extend([0.0, 0.0, 0.5])

        # Temporal Features (Padded Returns)
        num_temporal_features = self.state_history_length - 1
        padded_returns = np.zeros(num_temporal_features)
        if len(self.vwap_history) > 1:
            returns = np.diff(np.array(self.vwap_history))
            # Fill the end of the array with the most recent returns
            padded_returns[-len(returns):] = returns

        # --- Assemble the final state vector ---
        final_state_flat = np.array(
            global_and_agent_features + exchange_features + padded_returns.tolist(),
            dtype=np.float32
        )

        return final_state_flat.reshape(self.num_state_features, 1)

