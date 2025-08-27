import gym
import numpy as np
import pandas as pd
from collections import deque
from typing import List, Dict, Tuple


class HistoricalTradingEnv_v02(gym.Env):
    """
    A Gym environment for training and testing trading agents on historical data.
    It reads a CSV of trade data and simulates the agent's interaction with the market.
    """

    def __init__(self, bg_params: dict, env_params: dict):
        super(HistoricalTradingEnv_v02, self).__init__()
        self.seed()

        self.bg_params = bg_params
        self.env_params = env_params

        self.starting_cash = bg_params.get('starting_cash', 10_000_000_000)
        self.num_exchanges = bg_params.get('exchange_params', {}).get('num_exchange_agents', 2)
        self.reward_mode = env_params.get("reward_mode", "dense")
        self.state_history_length = env_params.get('state_history_length', 5)
        self.debug_mode = env_params.get('debug_mode', False)


        self.use_confidence_sizing = env_params.get('use_confidence_sizing', False)
        self.min_trade_size = 10
        self.max_trade_size = 250
        self.order_fixed_size = 100

        # Withdrawal fee options
        self.withdrawal_fees_enabled = bg_params.get('withdrawal_fees_enabled', True)
        self.withdrawal_fee_multiplier = bg_params.get('withdrawal_fee_multiplier', 15)


        # Transfer modelling
        self.min_transfer_delay_minutes: int = 5
        self.max_transfer_delay_minutes: int = 70
        self.transfer_delay_alpha: float = 5.0
        self.transfer_delay_beta: float = 14.5


        # Define Action and Observation Spaces (to match ABIDES env)
        self.num_actions = self.num_exchanges ** 2 + self.num_exchanges + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.features_per_exchange = 3
        self.num_global_features = 7
        self.num_temporal_features = self.state_history_length - 1
        self.num_state_features = (
                                              self.num_exchanges * self.features_per_exchange) + self.num_global_features + self.num_temporal_features

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_state_features, 1), dtype=np.float32
        )

        self.dfs = []
        self.max_steps = 0

        # Initialize Internal State
        self.current_step = 0
        self.cash = self.starting_cash
        self.total_holdings = 0
        self.realised_pnl = 0.0
        self.holdings_by_exchange = [0] * self.num_exchanges
        self.latest_marked_to_market = float(self.starting_cash)
        self.vwap_history = deque(maxlen=self.state_history_length)
        self.pending_transfers = []


    def _load_data(self, data_paths: List[str]):
        """
        Loads and aligns historical data for multiple exchanges.
        with at least the following columns: 'timestamp', 'price', 'volume', 'vwap', 'buy_volume'.
        """

        if not data_paths or not isinstance(data_paths, list):
            raise ValueError("Historical environment requires a list of 'data_paths' to be provided.")
        print(data_paths, self.num_exchanges)
        assert len(data_paths) == self.num_exchanges, \
            "The number of data paths provided does not match num_exchange_agents in the config."

        self.dfs = []
        for path in data_paths:
            print(f"Loading historical data from: {path}")
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            self.dfs.append(df)

        # Align all dataframes to a common index to handle missing data points.
        master_index = self.dfs[0].index
        for df in self.dfs[1:]:
            master_index = master_index.union(df.index)

        aligned_dfs = []
        for df in self.dfs:
            # Reindex and fill any missing values from the previous valid observation.
            aligned_dfs.append(df.reindex(master_index).ffill().bfill())

        self.dfs = aligned_dfs
        self.max_steps = len(master_index)
        print(f"Loaded and aligned data for {self.num_exchanges} exchanges with {self.max_steps} steps.")

    def reset(self, override_bg_params: dict = None):
        """Resets the environment for a new episode."""
        # Start with a fresh copy of the initial background parameters
        current_bg_params = self.bg_params.copy()

        # Update with any overrides for this specific episode (e.g., new data_paths)
        if override_bg_params:
            current_bg_params.update(override_bg_params)

        # Use the updated parameters to configure the episode
        data_paths = current_bg_params.get('data_paths')
        if data_paths:
            self._load_data(data_paths)
        else:
            # This error is important for debugging configuration issues.
            raise ValueError("Could not find 'data_paths' in the final configuration for this episode.")

        self.current_step = 0
        self.cash = self.starting_cash
        self.total_holdings = 0
        self.realised_pnl = 0.0

        self.holdings_by_exchange = [0] * self.num_exchanges
        self.latest_marked_to_market = float(self.starting_cash)
        self.vwap_history = deque(maxlen=self.state_history_length)
        self.pending_transfers = []
        return self._calculate_state()

    def seed(self, seed=None):
        """Seeds the environment's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_transfer_time(self) -> int:
        """
        Uses ABIDES logic ---
        Generates a random transfer delay in simulation steps, skewed by a Beta distribution.
        Assumes 1 step = 1 minute, matching the typical historical data frequency.
        """
        sample = self.np_random.beta(self.transfer_delay_alpha, self.transfer_delay_beta)

        delay_range = self.max_transfer_delay_minutes - self.min_transfer_delay_minutes
        delay_minutes = self.min_transfer_delay_minutes + sample * delay_range

        return int(round(delay_minutes))

    def _process_pending_transfers(self):
        """Checks the transfer queue and completes any transfers due by the current step."""
        completed_this_step = []
        for transfer in self.pending_transfers:
            if self.current_step >= transfer['completion_step']:
                # Transfer is complete, add assets to the destination exchange
                self.holdings_by_exchange[transfer['to_exchange']] += transfer['size']
                completed_this_step.append(transfer)
                if self.debug_mode:
                    print(
                        f"--- Step {self.current_step}: Transfer of {transfer['size']} shares to Exchange {transfer['to_exchange']} completed. ---")

        # Remove completed transfers from the pending list
        self.pending_transfers = [t for t in self.pending_transfers if t not in completed_this_step]

    def step(self, action_or_tuple: any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Executes one step, routing actions to the correct exchange."""
        self._process_pending_transfers()
        # check which mode the simulation is in
        action, confidence = (0, 0.0)
        if isinstance(action_or_tuple, tuple):
            action, confidence = action_or_tuple
        else:
            action = action_or_tuple

        assert self.action_space.contains(action), f"Action {action} is not contained in Action Space"

        #  Calculate trade size based on the mode
        if self.use_confidence_sizing:
            trade_size = self.min_trade_size + (self.max_trade_size - self.min_trade_size) * confidence
            trade_size = int(round(trade_size))
        else:
            trade_size = self.order_fixed_size

        current_prices = [df.iloc[self.current_step]['price'] for df in self.dfs]

        # Deconstruct and Execute Action
        # Action space for 2 exchanges: [0:HOLD, 1:BUY_E0, 2:SELL_E0, 3:BUY_E1, 4:SELL_E1, 5:TFR_1->0, 6:TFR_0->1]
        if 1 <= action <= self.num_exchanges * 2:
            exchange_id = (action - 1) // 2
            is_buy = (action - 1) % 2 == 0
            price = current_prices[exchange_id]

            if is_buy:
                cost = price * trade_size
                if self.cash >= cost:
                    self.cash -= cost
                    self.holdings_by_exchange[exchange_id] += trade_size
                    self.realised_pnl -= cost
            else:  # is SELL
                if self.holdings_by_exchange[exchange_id] >= trade_size:
                    self.cash += price * trade_size
                    self.holdings_by_exchange[exchange_id] -= trade_size
                    self.realised_pnl += price * trade_size

        elif self.num_exchanges == 2 and action in [5, 6]:
            from_exchange = 1 if action == 5 else 0
            to_exchange = 0 if action == 5 else 1

            if self.holdings_by_exchange[from_exchange] >= trade_size:
                # Immediately remove holdings from the source exchange
                self.holdings_by_exchange[from_exchange] -= trade_size

                if self.withdrawal_fees_enabled:
                    # Use the price on the source exchange to calculate the fee
                    fee = self.withdrawal_fee_multiplier * current_prices[from_exchange]
                    if self.cash >= fee:
                        self.cash -= fee
                    else:
                        # Not enough cash for the fee, so the transfer fails
                        # This implicitly penalizes the agent for trying an unaffordable action
                        pass

                # Schedule the completion of the transfer
                delay_steps = self._get_transfer_time()
                completion_step = self.current_step + delay_steps
                self.pending_transfers.append({
                    'to_exchange': to_exchange,
                    'size': trade_size,
                    'completion_step': completion_step
                })
                # if self.debug_mode:
                #     print(
                #         f"--- Step {self.current_step}: Initiated transfer of {trade_size} shares from E{from_exchange} to E{to_exchange}. ETA: Step {completion_step}. ---")

        # Advance the market
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        current_prices = [df.iloc[self.current_step]['price'] for df in self.dfs]
        price_map = {i: price for i, price in enumerate(current_prices)}

        # Create a mock holdings_by_exchange. In this simplified environment,
        # we assume all holdings are on a single "virtual" exchange (ID 0).
        mock_holdings_by_exchange = {
            i: {"ABM": holdings} for i, holdings in enumerate(self.holdings_by_exchange)
        }

        # Calculate the current portfolio value using the sophisticated method.
        current_portfolio_value = self._calculate_true_m2m(
            cash=self.cash,
            holdings_by_exchange=mock_holdings_by_exchange,
            price_map=price_map
        )

        # Calculate the dense reward (if applicable).
        reward = 0.0
        if self.reward_mode == "dense":
            reward = current_portfolio_value - self.latest_marked_to_market
            # Normalize the reward, just like in the ABIDES environment.
            if self.latest_marked_to_market > 0:
                reward = reward / self.latest_marked_to_market
            # Update the previous value for the next step.
            self.latest_marked_to_market = current_portfolio_value

        # Calculate the sparse reward at the end of the episode (if applicable).
        if done and self.reward_mode == "sparse":
            pnl = current_portfolio_value - self.starting_cash
            if pnl < 0:
                scaled_loss = pnl / self.starting_cash
                reward = - (scaled_loss ** 2)
            else:
                reward = pnl / self.starting_cash

        # Calculate New State and Info
        next_state = self._calculate_state()
        info = self._get_info()
        info['true_marked_to_market'] = current_portfolio_value
        info['realised_pnl'] = self.realised_pnl
        info['market_price'] = current_prices[0]

        return next_state, reward, done, info

    def _calculate_state(self) -> np.ndarray:
        """Calculates the state vector using data from all exchanges."""
        current_data_per_exchange = [df.iloc[self.current_step] for df in self.dfs]

        # Aggregate Global Features
        total_market_volume = sum(d.get('volume', 0) for d in current_data_per_exchange)
        total_buy_volume = sum(d.get('buy_volume', 0) for d in current_data_per_exchange)

        # Calculate global_vwap as the volume-weighted average of the per-exchange VWAPs
        sum_vwap_vol = sum(d.get('vwap', 0) * d.get('volume', 0) for d in current_data_per_exchange)
        global_vwap = sum_vwap_vol / total_market_volume if total_market_volume > 0 else 0
        self.vwap_history.append(global_vwap)

        # Volatility Calculation
        if len(self.vwap_history) > 1:
            log_returns = np.log(np.array(list(self.vwap_history))[1:] / np.array(list(self.vwap_history))[:-1])
            global_volatility = np.std(log_returns)
        else:
            global_volatility = 0

        # TVI as it requires buy/sell volume breakdown
        global_tvi = total_buy_volume / total_market_volume if total_market_volume > 0 else 0.5

        # Agent-Specific Features
        self.total_holdings = sum(self.holdings_by_exchange)
        pnl = self.latest_marked_to_market - self.starting_cash

        # Per-Exchange Features
        exchange_features = []
        for i in range(self.num_exchanges):
            local_data = current_data_per_exchange[i]
            local_vwap = local_data.get('vwap', local_data.get('price', 0))
            local_volume = local_data.get('volume', 0)
            local_buy_volume = local_data.get('buy_volume', 0)

            price_dev = local_vwap - global_vwap if local_vwap > 0 else 0
            vol_share = local_volume / total_market_volume if total_market_volume > 0 else (1 / self.num_exchanges)
            tvi = local_buy_volume / local_volume if local_volume > 0 else 0.5

            exchange_features.extend([price_dev, vol_share, tvi])

        # Temporal Features (from global VWAP history)
        padded_returns = np.zeros(self.num_temporal_features)
        if len(self.vwap_history) > 1:
            vwap_array = np.array(list(self.vwap_history))
            returns = np.log(vwap_array[1:] / vwap_array[:-1])
            padded_returns[-len(returns):] = returns

        # Assemble final state vector
        final_state_flat = np.array(
            [global_vwap, total_market_volume, global_tvi, global_volatility, self.cash, self.total_holdings, pnl] +
            exchange_features +
            padded_returns.tolist(),
            dtype=np.float32
        )
        return final_state_flat.reshape(self.num_state_features, 1)

    def _calculate_true_m2m(
            self,
            cash: float,
            holdings_by_exchange: Dict[int, Dict[str, int]],
            price_map: Dict[int, float],
            withdrawal_fees: Dict[int, Dict[str, float]] = None
    ) -> float:
        """
        Calculates the true mark-to-market value by finding the best net
        liquidation value for all assets. This logic is identical to the
        ABIDES environment.
        """
        if withdrawal_fees is None:
            withdrawal_fees = {}

        total_value = cash

        # Iterate through each asset holding on each exchange
        for exchange_id, holdings in holdings_by_exchange.items():
            for symbol, shares in holdings.items():
                if shares == 0:
                    continue

                # Find the best possible net price for this specific asset
                best_net_price = 0

                # Check the price on every potential market
                for potential_market_id in range(self.num_exchanges):
                    market_price = price_map.get(potential_market_id, 0)
                    if market_price == 0:
                        continue

                    # In the historical env, we assume no transfer costs.
                    net_price = market_price
                    if net_price > best_net_price:
                        best_net_price = net_price

                # Add the asset's true value to the total
                total_value += shares * best_net_price
        return total_value

    def _get_info(self) -> Dict:
        """Returns diagnostic information, including pending transfers."""
        if not self.debug_mode: return {}

        current_prices = {f'price_ex_{i}': df.iloc[self.current_step]['price'] for i, df in enumerate(self.dfs)}

        info_dict = {
            'step': self.current_step,
            'cash': self.cash,
            'total_holdings': sum(self.holdings_by_exchange),
            'holdings_by_exchange': self.holdings_by_exchange,
            'true_marked_to_market': self.latest_marked_to_market,
            'pending_transfers': self.pending_transfers  # Add this for debugging
        }
        info_dict.update(current_prices)
        return info_dict