import importlib
from typing import Any, Dict, List, Union
from collections import deque

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv


class SubGymMarketsCryptoDailyInvestorEnv_v02(AbidesGymMarketsEnv):
    """
    A refactored Gym environment for a multi-exchange simulation.

    This environment correctly models a world with multiple exchanges, segregated
    asset holdings, and withdrawal fees. The action and observation spaces have
    been expanded to allow the RL agent to make decisions across all markets.
    """

    # Decorators remain the same
    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )
    raw_state_pre_process_multi = markets_agent_utils.keep_last_by_exchange_decorator

    def __init__(
            self,
            background_config: Union[str, Dict] = "cdormsc02",
            mkt_close: str = "23:59:59",
            timestep_duration: str = "60s",
            starting_cash: int = 1_000_000,
            order_fixed_size: int = 10,
            state_history_length: int = 4,
            market_data_buffer_length: int = 5,
            first_interval: str = "00:05:00",
            reward_mode: str = "dense",
            done_ratio: float = 0.3,
            debug_mode: bool = False,
            background_config_extra_kvargs={}
    ) -> None:

        # Handle both config types (dict or str) Dict for yaml jobs and str from traditional execution
        config_callable = None
        config_args = {"end_time": mkt_close}
        config_args.update(background_config_extra_kvargs)

        if isinstance(background_config, str):
            # If a string is passed, import it as a module name (the old way).
            assert background_config in [
                "cdormsc01", "cdormsc02"
            ], "Select one of cdormsc01, cdormsc02 as config"
            config_module = importlib.import_module(f"gym_crypto_markets.configs.{background_config}")
            config_callable = config_module.build_config
            self.config = config_callable(**config_args)



        elif isinstance(background_config, dict):
            # If a dictionary is passed, use it directly (the new, flexible way).
            print("INFO: Using pre-built dictionary for background_config.")
            self.config = background_config
            # The parent class expects a function, so we create a simple lambda
            # that just returns our pre-built dictionary.
            config_callable = lambda **kwargs: self.config
            config_args = {}
        else:
            raise TypeError(f"background_config must be a string or a dict, but got {type(background_config)}")

        # self.background_config_module: Any = importlib.import_module(f"gym_crypto_markets.configs.{background_config}")
        # background_config_args = {"end_time": mkt_close}
        # background_config_args.update(background_config_extra_kvargs)
        # self.config = self.background_config_module.build_config(**background_config_args)
        self.num_exchanges = self.config.get("num_exchange_agents", 1)
        self.mkt_open: NanosecondTime = self.config.get("mkt_open", str_to_ns("00:10:00"))
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.reward_mode: str = reward_mode
        self.done_ratio: float = done_ratio
        self.debug_mode: bool = debug_mode

        # Time Normalisation for M2M: Calculate the number of steps efficiently using these dynamic values
        trading_day_duration = self.mkt_close - self.mkt_open
        if self.timestep_duration > 0 and trading_day_duration > 0:
            self.num_steps_per_episode = trading_day_duration / self.timestep_duration
        else:
            self.num_steps_per_episode = 1

        # marked_to_market limit to STOP the episode
        self.down_done_condition: float = self.done_ratio * starting_cash

        # CHECK PROPERTIES

        assert (self.first_interval <= str_to_ns("16:00:00")) & (
                self.first_interval >= str_to_ns("00:00:00")
        ), "Select authorized FIRST_INTERVAL delay"

        assert (self.mkt_close <= str_to_ns("23:59:59")) & (
                self.mkt_close >= str_to_ns("00:00:00")
        ), "Select authorized market hours"

        assert reward_mode in [
            "sparse",
            "dense",
        ], "reward_mode needs to be dense or sparse"

        assert (self.timestep_duration <= str_to_ns("06:30:00")) & (
                self.timestep_duration >= str_to_ns("00:00:00")
        ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
                self.starting_cash >= 0
        ), "Select positive integer value for starting_cash"

        assert (type(self.order_fixed_size) == int) & (
                self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.state_history_length) == int) & (
                self.state_history_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.market_data_buffer_length) == int) & (
                self.market_data_buffer_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (
                (type(self.done_ratio) == float)
                & (self.done_ratio >= 0)
                & (self.done_ratio < 1)
        ), "Select positive float value for order_fixed_size between 0 and 1"

        assert debug_mode in [
            True,
            False,
        ], "reward_mode needs to be True or False"

        # Pass config to the parent AbidesGymMarketsEnv
        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(config_callable, config_args),
            wakeup_interval_generator=ConstantTimeGenerator(step_duration=self.timestep_duration),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,  # Now using the instance attribute
            market_data_buffer_length=self.market_data_buffer_length,  # Now using the instance attribute
            first_interval=self.first_interval,
        )

        # --- Action Space ---
        # The action space is now (2 * num_exchanges) + 1 to account for
        # TFR between each pair, BUY/SELL on each exchange, plus a single HOLD action.
        # TFR = E(E-1), BUY/SELL = 2E and HOLD = 1, therefore together = E^2 + E + 1
        self.num_actions: int = (self.num_exchanges)**2 + self.num_exchanges + 1
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # --- Observation Space ---
        # The state now includes features for each exchange, plus global features.
        # Features per exchange: [Imbalance, Spread, DirectionFeature]
        # Global features: [Total Holdings]
        self.features_per_exchange = 3  # Imbalance, Spread, DirectionFeature
        self.num_global_features = 1  # Total Holdings
        self.num_temporal_features = self.state_history_length - 1  # Padded Returns
        self.num_state_features: int = (
                (self.num_exchanges * self.features_per_exchange) +
                self.num_global_features +
                self.num_temporal_features
        )

        # An attribute to store the history of global mid-prices
        self.global_mid_price_history = deque(maxlen=self.state_history_length)

        # Define the high and low bounds as flat arrays first
        state_highs_flat = np.array(
            [np.finfo(np.float32).max] +  # Total Holdings
            self.num_exchanges * [
                1.0,  # Imbalance
                np.finfo(np.float32).max,  # Spread
                np.finfo(np.float32).max,  # DirectionFeature
            ] +
            self.num_temporal_features * [np.finfo(np.float32).max],  # Padded Returns
            dtype=np.float32,
        )
        state_lows_flat = np.array(
            [np.finfo(np.float32).min] +  # Total Holdings
            self.num_exchanges * [
                0.0,  # Imbalance
                np.finfo(np.float32).min,  # Spread
                np.finfo(np.float32).min,  # DirectionFeature
            ] +
            self.num_temporal_features * [np.finfo(np.float32).min],  # Padded Returns
            dtype=np.float32,
        )

        # Apply .reshape() to create column vectors, as per the original convention.
        self.state_highs: np.ndarray = state_highs_flat.reshape(self.num_state_features, 1)
        self.state_lows: np.ndarray = state_lows_flat.reshape(self.num_state_features, 1)

        # Explicitly define the shape for the observation space.
        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32,
        )

        self.previous_marked_to_market = self.starting_cash

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action: int) -> List[Dict[str, Any]]:
        """
        --- Maps the new, larger action space to ABIDES actions ---
        The action now implies both an operation (BUY/SELL/HOLD) and a target exchange.
        Example for 2 exchanges:
        - 0: HOLD (i.e. do nothing)
        - 1: BUY on Exchange 0
        - 2: SELL on Exchange 0
        - 3: BUY on Exchange 1
        - 4: SELL on Exchange 1
        - 5: TRANSFER_FROM_0_TO_1
        - 6: TRANSFER_FROM_1_TO_0
        """
        if action == 0:
            return []  # HOLD

        # --- Action 1-4: BUY/SELL on exchanges
        if 1 <= action <= 4:
            # Deconstruct the action into an exchange and a direction
            action_index = action - 1
            exchange_id = action_index // 2
            direction = "BUY" if (action_index % 2) == 0 else "SELL"

            return [{
                "type": "MKT",
                "direction": direction,
                "size": self.order_fixed_size,
                "exchange_id": exchange_id
            }]
        # --- Action 5: TRANSFER_FROM_0_TO_1 ---
        elif action == 5:
            return [{
                "type": "TFR",
                "from_exchange": 1,
                "to_exchange": 0,
                "size": self.order_fixed_size,
            }]

        # --- Action 6: TRANSFER_FROM_1_TO_0 ---

        elif action == 6:
            return [{
                "type": "TFR",
                "from_exchange": 0,
                "to_exchange": 1,
                "size": self.order_fixed_size,
            }]

        else:
            print(f"Unknown action {action} received. Defaulting to HOLD.")
            return []

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Constructs the new state vector from multi-exchange data ---
        """
        mkt_data_buffer = raw_state.get("parsed_mkt_data", {})
        internal_data = raw_state.get("internal_data", {})

        # 1. Get Global Feature: Total Holdings
        # This is already calculated correctly by the parent agent.
        total_holdings = internal_data.get("total_holdings", 0)[-1]

        # 2. Calculate Per-Exchange Features
        per_exchange_features = []
        latest_data_by_exchange = {}

        exchange_ids = mkt_data_buffer.get('exchange_id', [])
        num_updates = len(exchange_ids)

        # Loop through the history in reverse to find the latest update for each exchange.
        for i in reversed(range(num_updates)):
            ex_id = mkt_data_buffer['exchange_id'][i]

            # If we haven't found data for this exchange yet, this must be the latest.
            if ex_id not in latest_data_by_exchange:
                latest_data_by_exchange[ex_id] = {
                    "bids": mkt_data_buffer['bids'][i],
                    "asks": mkt_data_buffer['asks'][i],
                    "last_transaction": mkt_data_buffer['last_transaction'][i]
                }
        for ex_id in range(self.num_exchanges):
            data = latest_data_by_exchange.get(ex_id)
            if data:
                bids, asks, last_trans = data["bids"], data["asks"], data["last_transaction"]
                imbalance = markets_agent_utils.get_imbalance(bids, asks, depth=3)
                spread = (asks[0][0] - bids[0][0]) if asks and bids else 0
                mid_price = (asks[0][0] + bids[0][0]) / 2 if asks and bids else last_trans
                direction = mid_price - last_trans
                per_exchange_features.extend([imbalance, spread, direction])
            else:
                per_exchange_features.extend([0.5, 0, 0])

        # Returns
        # 3. Calculate Global Mid-Price and Historical Returns
        best_global_bid, best_global_ask = -1, float('inf')
        for ex_id, data in latest_data_by_exchange.items():
            bids, asks = data["bids"], data["asks"]
            if bids and bids[0][0] > best_global_bid:
                best_global_bid = bids[0][0]
            if asks and asks[0][0] < best_global_ask:
                best_global_ask = asks[0][0]

        # Calculate global mid-price if a valid spread exists across any exchange.
        if best_global_bid != -1 and best_global_ask != float('inf'):
            global_mid_price = (best_global_bid + best_global_ask) / 2
            self.global_mid_price_history.append(global_mid_price)

        # Compute returns from the history of global mid-prices.
        returns = np.diff(list(self.global_mid_price_history)) if len(self.global_mid_price_history) > 1 else np.array([])

        # Pad the returns to ensure the state vector always has a fixed size.

        padded_returns = np.zeros(self.state_history_length - 1)
        if len(returns) > 0:
            padded_returns[-len(returns):] = returns

        # 4. Assemble the final state vector.
        final_state_flat = np.array(
            [total_holdings] + per_exchange_features + padded_returns.tolist(),
            dtype=np.float32
        )

        # 5. Reshape to a column vector to match the environment's convention.
        return final_state_flat.reshape(self.num_state_features, 1)

    def _calculate_true_m2m(self, raw_state: List[Any]) -> float:
        """
        Calculates the true mark-to-market value by finding the best net
        liquidation value for all assets, accounting for withdrawal fees.
        """
        processed_state = raw_state[0]
        mkt_data = processed_state.get("parsed_mkt_data", [])

        internal_data = processed_state["internal_data"]
        cash = internal_data.get("cash", 0)
        holdings_by_exchange = internal_data.get("holdings_by_exchange", {})
        withdrawal_fees = internal_data.get("withdrawal_fees", {})

        total_value = cash

        price_map = {
            d["exchange_id"]: d.get("last_transaction", 0)
            for d in mkt_data if "exchange_id" in d
        }

        # Iterate through each asset holding on each exchange
        for exchange_id, holdings in holdings_by_exchange.items():
            for symbol, shares in holdings.items():
                if shares == 0:
                    continue

                # Find the best possible net price for this specific asset
                best_net_price = 0

                # Check the price on every potential market
                for potential_market_id in range(self.num_exchanges):
                    # Find the latest price on this potential market
                    market_price = price_map.get(potential_market_id, 0)

                    if market_price == 0:
                        continue

                    # Calculate the cost to move the asset to this market
                    fee_structure = withdrawal_fees.get(exchange_id, {})
                    transfer_cost = 0
                    if potential_market_id != exchange_id:
                        transfer_cost = fee_structure.get(symbol, fee_structure.get('default', 0))

                    net_price = market_price - transfer_cost

                    if net_price > best_net_price:
                        best_net_price = net_price

                # Add the asset's true value to the total
                total_value += shares * best_net_price
        return total_value

    @raw_state_pre_process_multi
    def raw_state_to_reward(self, raw_state: List[Any]) -> float:
        """
        --- Uses the new robust M2M calculation ---
        """
        if self.reward_mode == "dense":
            marked_to_market = self._calculate_true_m2m(raw_state)
            reward = marked_to_market - self.previous_marked_to_market
            self.previous_marked_to_market = marked_to_market

            # Normalize reward (optional, but good practice)
            if self.order_fixed_size > 0:
                reward = reward / self.order_fixed_size  # Normalize by a typical trade value
            if self.num_steps_per_episode > 0:
                reward = reward / self.num_steps_per_episode
            return reward
        elif self.reward_mode == "sparse":
            return 0

    @raw_state_pre_process_multi
    def raw_state_to_update_reward(self, raw_state: List[Any]) -> float:
        """Calculates the sparse reward (total episode gain/loss) at the end of an episode."""
        if self.reward_mode == "dense":
            return 0
        elif self.reward_mode == "sparse":
            marked_to_market = self._calculate_true_m2m(raw_state)
            reward = marked_to_market - self.starting_cash
            if self.order_fixed_size > 0: reward /= self.order_fixed_size
            if self.num_steps_per_episode > 0: reward /= self.num_steps_per_episode
            return reward
    @raw_state_pre_process_multi
    def raw_state_to_done(self, raw_state: List[Any]) -> bool:
        """Determines if the episode is done by checking the agent's portfolio value."""
        marked_to_market = self._calculate_true_m2m(raw_state)
        return marked_to_market <= self.down_done_condition

    @raw_state_pre_process_multi
    def raw_state_to_info(self, raw_state: List[Any]) -> Dict[str, Any]:
        """
        Transforms a raw state into a rich, structured dictionary for debugging.
        This version is adapted for a multi-exchange environment.
        """
        if not self.debug_mode:
            return {}

        processed_state = raw_state[0]

        # Extract the core data structures from the raw state.
        mkt_data = processed_state["parsed_mkt_data"]
        internal_data = processed_state["internal_data"]

        #print(mkt_data)

        # --- Global Agent Information ---
        info = {
            "current_time": internal_data.get("current_time"),
            "cash": internal_data.get("cash"),
            "holdings_by_exchange": internal_data.get("holdings_by_exchange", 0),
            "withdrawal_fees": internal_data.get("withdrawal_fees", {}),
            "total_holdings": internal_data.get("total_holdings", 0),
            "order_status": internal_data.get("order_status"),
            # Pass the processed state to the calculation function
            "true_marked_to_market": self._calculate_true_m2m(raw_state),
            "mkt_open_times": internal_data.get("mkt_open_times"),
            "mkt_close_times": internal_data.get("mkt_close_times"),
        }

        # --- Per-Exchange Market Information ---
        exchange_data = {}
        for latest_data in mkt_data:
            ex_id = latest_data.get("exchange_id")
            if ex_id is None:
                continue

            bids = latest_data.get("bids", [])
            asks = latest_data.get("asks", [])
            last_trans = latest_data.get("last_transaction", 0)

            best_bid = bids[0][0] if bids else last_trans
            best_ask = asks[0][0] if asks else last_trans

            last_bid = markets_agent_utils.get_last_val(bids, last_trans)
            last_ask = markets_agent_utils.get_last_val(asks, last_trans)

            orderbook = {"asks": {"price": {}, "volume": {}}, "bids": {"price": {}, "volume": {}}}
            for book, book_name in [(bids, "bids"), (asks, "asks")]:
                for level in [0, 1, 2]:
                    price, volume = markets_agent_utils.get_val(book, level)
                    orderbook[book_name]["price"][level] = np.array([price]).reshape(-1)
                    orderbook[book_name]["volume"][level] = np.array([volume]).reshape(-1)

            exchange_data[ex_id] = {
                "last_transaction": last_trans,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": best_ask - best_bid,
                "detailed_orderbook": orderbook,
                "last_bid_in_book": last_bid,
                "last_ask_in_book": last_ask,
                "wide_spread": last_ask - last_bid,
            }

        for ex_id in range(self.num_exchanges):
            if ex_id not in exchange_data:
                exchange_data[ex_id] = {"status": "No data received"}

        info["market_data"] = exchange_data
        #print(info)
        return info