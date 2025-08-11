from copy import deepcopy
import importlib
from typing import Any, Dict, List, Union, Tuple
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
            starting_cash: int = 10_000_000_000,
            use_confidence_sizing: bool = False,
            order_fixed_size: int = 100,
            state_history_length: int = 5,
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
        self.use_confidence_sizing: bool = use_confidence_sizing
        self.min_trade_size: int = 10
        self.max_trade_size: int = 250
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.reward_mode: str = reward_mode
        self.done_ratio: float = done_ratio
        self.debug_mode: bool = debug_mode
        self.latest_price: Dict[int, int] = {}

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
        # Features per exchange: [Price deviation, Vol share, TVI]
        # Global features: [Global VWAP, Global TVI, Global Market Vol, Global volatility ]
        # Agent features: [Total Holdings, Cash, PnL, Total Returns(1s, 5s, 1m, 5m]
        self.features_per_exchange = 3  # Imbalance, Spread, DirectionFeature
        self.num_global_features = 7  # Global + Agent features
        self.num_temporal_features = self.state_history_length - 1  # Padded Returns
        self.num_state_features: int = (
                (self.num_exchanges * self.features_per_exchange) +
                self.num_global_features +
                self.num_temporal_features
        )

        # An attribute to store the history of global mid-prices
        self.global_mid_price_history = deque(maxlen=self.state_history_length)
        self.aggregate_history = deque(maxlen=10)


        # Define the high and low bounds as flat arrays first
        state_highs_flat = np.array(
            # Global features
            [
                np.finfo(np.float32).max, # VWAP
                np.finfo(np.float32).max,  # Total Market Volume
                1.0,                       # Global Trade Imbalance
                np.finfo(np.float32).max,  # Overall Volatility
                np.finfo(np.float32).max,  # Cash
                np.finfo(np.float32).max,  # Total Holdings
                np.finfo(np.float32).max,  # Pnl

            ] +
            # Per exchange features
            self.num_exchanges * [
                np.finfo(np.float32).max,  # Price Deviation
                1.0,                       # Volume Share
                1.0,                       # TVI per exchange
            ] +
            # Temporal features
            self.num_temporal_features * [np.finfo(np.float32).max],  # Padded Returns
            dtype=np.float32,
        )
        state_lows_flat = np.array(
            # Global Features
            [
                0.0,                       # VWAP
                0.0,                       # Total Market Volume
                0.0,                       # Global Trade Imbalance
                0.0,                       # Overall Volatility
                0.0,                       # Cash
                np.finfo(np.float32).min,  # Holdings (can be short)
                np.finfo(np.float32).min,  # PnL
            ] +
            # Per-Exchange Features
            self.num_exchanges * [
                np.finfo(np.float32).min,  # Price Deviation
                0.0,                       # Volume Share
                0.0,                       # TVI per exchange
            ] +
            # Temporal Features
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
        self.realised_pnl = 0.0

    def reset(self, override_bg_params: dict = None):
        """
        Overrides the parent reset method to ensure all custom state variables
        are properly re-initialized for each new episode.
        """
        # First, call the parent's reset method to handle all the core
        # ABIDES simulation and kernel resetting. This returns the initial state.
        initial_state = super().reset()

        # Now, reset the custom attributes for this child environment.
        self.realised_pnl = 0.0
        self.previous_marked_to_market = float(self.starting_cash)
        self.aggregate_history.clear()
        self.global_mid_price_history.clear()
        return initial_state

    def step(self, action_or_tuple: any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        action, confidence = (0, 0.0)  # Default values

        if isinstance(action_or_tuple, tuple):
            # In confidence mode, we expect a tuple
            action, confidence = action_or_tuple
        else:
            # In fixed mode, we expect a single integer
            action = action_or_tuple

        assert self.action_space.contains(
            action
        ), f"Action {action} is not contained in Action Space"

        abides_action = self._map_action_space_to_ABIDES_SIMULATOR_SPACE(action, confidence)

        raw_state = self.kernel.runner((self.gym_agent, abides_action))
        self.state = self.raw_state_to_state(deepcopy(raw_state["result"]))

        assert self.observation_space.contains(
            self.state
        ), f"INVALID STATE {self.state}"

        self.reward = self.raw_state_to_reward(deepcopy(raw_state["result"]))
        self.done = raw_state["done"] or self.raw_state_to_done(
            deepcopy(raw_state["result"])
        )

        if self.done:
            self.reward += self.raw_state_to_update_reward(
                deepcopy(raw_state["result"])
            )

        self.info = self.raw_state_to_info(deepcopy(raw_state["result"]))

        return (self.state, self.reward, self.done, self.info)

    def _get_latest_ask_price(self, exchange_id: int) -> float:
        """
        Helper function to get the best ask price for a given exchange from the agent's memory.
        This represents the most recent price information the agent has before acting.
        """
        if hasattr(self.gym_agent, 'parsed_mkt_data_buffer'):
            for data in reversed(self.gym_agent.parsed_mkt_data_buffer):
                if data.get("exchange_id") == exchange_id:
                    asks = data.get("asks", [])
                    if asks and len(asks) > 0 and len(asks[0]) > 0:
                        return asks[0][0]

        if hasattr(self.gym_agent, 'get_last_trade'):
            last_trade_price = self.gym_agent.get_last_trade(exchange_id, "ABM")
            if last_trade_price is not None and last_trade_price > 0:
                return last_trade_price

        return 0.0
    def _get_latest_bid_price(self, exchange_id: int) -> float:
        """
        Helper function to get the best bid price for a given exchange from the agent's memory.
        This represents the most recent price information the agent has before acting.
        """
        if hasattr(self.gym_agent, 'parsed_mkt_data_buffer'):
            for data in reversed(self.gym_agent.parsed_mkt_data_buffer):
                if data.get("exchange_id") == exchange_id:
                    bids = data.get("bids", [])
                    if bids and len(bids) > 0 and len(bids[0]) > 0:
                        return bids[0][0]

        if hasattr(self.gym_agent, 'get_last_trade'):
            last_trade_price = self.gym_agent.get_last_trade(exchange_id, "ABM")
            if last_trade_price is not None and last_trade_price > 0:
                return last_trade_price

        return 0.0

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action: int, confidence: float) -> List[Dict[str, Any]]:
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
        if self.use_confidence_sizing:
            trade_size = self.min_trade_size + (self.max_trade_size - self.min_trade_size) * confidence
            trade_size = int(round(trade_size))
        else:
            trade_size = self.order_fixed_size

        # --- Action 1-4: BUY/SELL on exchanges
        if 1 <= action <= 4:
            # Deconstruct the action into an exchange and a direction
            action_index = action - 1
            exchange_id = action_index // 2
            is_buy = (action_index % 2) == 0
            if is_buy:
                current_ask_price = self._get_latest_ask_price(exchange_id)
                estimated_cost = current_ask_price * trade_size

                if self.gym_agent.cash < estimated_cost:

                    print(f"INFO: Agent tried to BUY with insufficient cash. Overiding HOLD")
                    return []
                else:
                    self.realised_pnl -= estimated_cost
            else:
                current_bid_price = self._get_latest_bid_price(exchange_id)
                self.realised_pnl += current_bid_price * trade_size

            direction = "BUY" if is_buy else "SELL"

            return [{
                "type": "MKT",
                "direction": direction,
                "size": trade_size,
                "exchange_id": exchange_id
            }]
        # --- Action 5: TRANSFER_FROM_0_TO_1 ---
        elif action in [5, 6] and self.num_exchanges > 1:
            if action == 5:
                from_exchange = 1
                to_exchange = 0
                holdings_on_source = self.gym_agent.holdings_by_exchange.get(from_exchange, {}).get("ABM", 0)
                self.realised_pnl -= self.gym_agent.withdrawal_fees.get(from_exchange).get("ABM", self.gym_agent.withdrawal_fees.get(from_exchange).get("default", 0))

                if holdings_on_source < trade_size:
                    print(
                        f"INFO: Agent tried to TFR {trade_size} from Exch {from_exchange} but only holds {holdings_on_source}. Overriding to HOLD.")
                    return []
                return [{
                        "type": "TFR",
                        "from_exchange": from_exchange,
                        "to_exchange": to_exchange,
                        "size": trade_size,
                    }]

            # --- Action 6: TRANSFER_FROM_1_TO_0 ---

            elif action == 6:
                from_exchange = 0
                to_exchange = 1

                holdings_on_source = self.gym_agent.holdings_by_exchange.get(from_exchange, {}).get("ABM", 0)
                self.realised_pnl -= self.gym_agent.withdrawal_fees.get(from_exchange).get("ABM", self.gym_agent.withdrawal_fees.get(from_exchange).get("default", 0))
                if holdings_on_source < trade_size:
                    print(
                        f"INFO: Agent tried to TFR {trade_size} from Exch {from_exchange} but only holds {holdings_on_source}. Overriding to HOLD.")
                    return []

                return [{
                        "type": "TFR",
                        "from_exchange": from_exchange,
                        "to_exchange": to_exchange,
                        "size": trade_size,
                    }]

            else:
                print(f"Unknown action {action} received. Defaulting to HOLD.")
                return []

    def _get_action_mask(self) -> np.ndarray:
        """
        Generates a mask of valid actions based on the agent's current state.
        Returns a numpy array where 1 indicates a valid action and 0 an invalid one
        """
        # The action space is: [0:HOLD, 1:BUY_E0, 2:SELL_E0, 3:BUY_E1, 4:SELL_E1, 5:TFR_1->0, 6:TFR_0->1]
        mask = np.ones(self.num_actions, dtype=np.int8)

        # Rule 1: Check cash for BUY actions
        # Actions 1 and 3 and BUY on exchanges 0 and 1 respectively
        if self.gym_agent.cash <= 0:
            mask[1] = 0 # Disable BUY on Exchange 0
            mask[3] = 0 # Disable BUY on Exchange 1
        if self.num_exchanges == 2:
            # Rule 2: Check holdings for transfer actions
            # action 5: TFR from 1 to 0
            holdings_on_exch1 = self.gym_agent.holdings_by_exchange.get(1, {}).get("ABM", 0)
            if holdings_on_exch1 < self.order_fixed_size:
                mask[5] = 0  # Disable TFR from Exchange 1

            # Action 6: TFR from 0 to 1
            holdings_on_exch0 = self.gym_agent.holdings_by_exchange.get(0, {}).get("ABM", 0)
            if holdings_on_exch0 < self.order_fixed_size:
                mask[6] = 0  # Disable TFR from Exchange 0
        else:
            if self.num_actions > 3: mask[3] = 0
            if self.num_actions > 4: mask[4] = 0
            if self.num_actions > 5: mask[5] = 0
            if self.num_actions > 6: mask[6] = 0

        return mask

    @raw_state_pre_process_multi
    def raw_state_to_state(self, raw_state: List[Any]) -> np.ndarray:
        """
            Orchestrates state creation:
            1. Aggregates new raw trades and adds them to a clean history.
            2. Wipes the raw trade buffer.
            3. Computes the final state vector from the clean aggregate history.
        """
        processed_state = raw_state[0]
        internal_data = processed_state.get("internal_data", {})

        new_trades = processed_state.get("parsed_trade_data", [])

        if new_trades:
            current_interval_summary = self.calculate_aggregates(new_trades)
            self.aggregate_history.append(current_interval_summary)

        # Calculate all features from the clean aggregate history ---
        total_sum_price_vol = sum(interval['sum_price_vol'] for interval in self.aggregate_history)
        total_traded_volume = sum(interval['total_volume'] for interval in self.aggregate_history)
        total_buy_volume = sum(interval['buy_volume'] for interval in self.aggregate_history)

        global_vwap = total_sum_price_vol / total_traded_volume if total_traded_volume > 0 else 0
        global_tvi = total_buy_volume / total_traded_volume if total_traded_volume > 0 else 0.5

        # Calculate volatility from the history of interval VWAPs
        interval_vwaps = [
            interval['sum_price_vol'] / interval['total_volume']
            for interval in self.aggregate_history if interval['total_volume'] > 0
        ]
        if len(interval_vwaps) > 1:
            log_returns = np.log(np.array(interval_vwaps[1:]) / np.array(interval_vwaps[:-1]))
            global_volatility = np.std(log_returns)
        else:
            global_volatility = 0

        global_market_features = [global_vwap, total_traded_volume, global_tvi, global_volatility]

        # Agent Features
        total_holdings = internal_data.get("total_holdings", 0)
        cash = internal_data.get("cash", 0)
        pnl = self._calculate_true_m2m(raw_state) - self.starting_cash
        agent_features = [cash, total_holdings, pnl]

        # Exchange-specific features from the most recent interval
        exchange_features = []
        most_recent_summary = self.aggregate_history[-1] if self.aggregate_history else {}
        most_recent_exchange_data = most_recent_summary.get('per_exchange_data', {})

        for ex_id in range(self.num_exchanges):
            ex_data = most_recent_exchange_data.get(ex_id, {})
            ex_total_volume = ex_data.get('total_volume', 0)

            local_vwap = ex_data.get('sum_price_vol', 0) / ex_total_volume if ex_total_volume > 0 else 0
            price_dev = local_vwap - global_vwap if local_vwap > 0 else 0
            vol_share = ex_total_volume / total_traded_volume if total_traded_volume > 0 else 0
            tvi = ex_data.get('buy_volume', 0) / ex_total_volume if ex_total_volume > 0 else 0.5
            exchange_features.extend([price_dev, vol_share, tvi])

        # Temporal features (returns) from the history of interval VWAPs
        relevant_vwaps = interval_vwaps[-self.state_history_length:]
        returns = np.diff(relevant_vwaps) if len(relevant_vwaps) > 1 else np.array([])

        padded_returns = np.zeros(self.num_temporal_features)
        if len(returns) > 0:
            padded_returns[-len(returns):] = returns

        global_and_agent_features = [
            global_vwap,
            total_traded_volume,
            global_tvi,
            global_volatility,
            cash,
            total_holdings,
            pnl
        ]

        # --- Step 4: Assemble the final state vector ---
        final_state_flat = np.array(
            global_and_agent_features + exchange_features + padded_returns.tolist(),
            dtype=np.float32
        )

        # Below is the old logic when using mid_prices
        # # Calculate Per-Exchange Features
        # per_exchange_features = []
        # latest_data_by_exchange = {}
        #
        # exchange_ids = mkt_data_buffer.get('exchange_id', [])
        # num_updates = len(exchange_ids)
        #
        # # Loop through the history in reverse to find the latest update for each exchange.
        # for i in reversed(range(num_updates)):
        #     ex_id = mkt_data_buffer['exchange_id'][i]
        #
        #     # If we haven't found data for this exchange yet, this must be the latest.
        #     if ex_id not in latest_data_by_exchange:
        #         latest_data_by_exchange[ex_id] = {
        #             "bids": mkt_data_buffer['bids'][i],
        #             "asks": mkt_data_buffer['asks'][i],
        #             "last_transaction": mkt_data_buffer['last_transaction'][i]
        #         }
        # for ex_id in range(self.num_exchanges):
        #     data = latest_data_by_exchange.get(ex_id)
        #     if data:
        #         bids, asks, last_trans = data["bids"], data["asks"], data["last_transaction"]
        #         imbalance = markets_agent_utils.get_imbalance(bids, asks, depth=3)
        #         spread = (asks[0][0] - bids[0][0]) if asks and bids else 0
        #         mid_price = (asks[0][0] + bids[0][0]) / 2 if asks and bids else last_trans
        #         direction = mid_price - last_trans
        #         per_exchange_features.extend([imbalance, spread, direction])
        #     else:
        #         per_exchange_features.extend([0.5, 0, 0])
        #
        # # Returns
        # # 3. Calculate Global Mid-Price and Historical Returns
        # best_global_bid, best_global_ask = -1, float('inf')
        # for ex_id, data in latest_data_by_exchange.items():
        #     bids, asks = data["bids"], data["asks"]
        #     if bids and bids[0][0] > best_global_bid:
        #         best_global_bid = bids[0][0]
        #     if asks and asks[0][0] < best_global_ask:
        #         best_global_ask = asks[0][0]
        #
        # # Calculate global mid-price if a valid spread exists across any exchange.
        # if best_global_bid != -1 and best_global_ask != float('inf'):
        #     global_mid_price = (best_global_bid + best_global_ask) / 2
        #     self.global_mid_price_history.append(global_mid_price)
        #
        # # Compute returns from the history of global mid-prices.
        # returns = np.diff(list(self.global_mid_price_history)) if len(self.global_mid_price_history) > 1 else np.array([])
        #
        # # Pad the returns to ensure the state vector always has a fixed size.
        #
        # padded_returns = np.zeros(self.state_history_length - 1)
        # if len(returns) > 0:
        #     padded_returns[-len(returns):] = returns
        #
        # # 4. Assemble the final state vector.
        # final_state_flat = np.array(
        #     [total_holdings] + per_exchange_features + padded_returns.tolist(),
        #     dtype=np.float32
        # )

        # 5. Reshape to a column vector to match the environment's convention.

        return final_state_flat.reshape(self.num_state_features, 1)

    def calculate_aggregates(self, trade_history: List[Dict]) -> Dict:
        """
            Processes a list of raw trades from a single time interval and returns
            a single dictionary summarizing that interval.
            """

        interval_summary = {
            'sum_price_vol': 0,
            'total_volume': 0,
            'buy_volume': 0,
            'per_exchange_data': {}
        }

        # --- Step 2: Loop through the raw trades to populate the summary ---
        for trade in trade_history:
            price = trade.get('price')
            volume = trade.get('quantity')  # Correct key is 'quantity'
            ex_id = trade.get('exchange_id')

            # Skip if data is incomplete
            if price is None or volume is None or ex_id is None:
                continue

            # Update global summaries
            interval_summary['sum_price_vol'] += price * volume
            interval_summary['total_volume'] += volume
            if trade.get('is_buy'):
                interval_summary['buy_volume'] += volume

            # Update per-exchange summaries
            if ex_id not in interval_summary['per_exchange_data']:
                interval_summary['per_exchange_data'][ex_id] = {
                    'sum_price_vol': 0, 'total_volume': 0, 'buy_volume': 0
                }

            ex_summary = interval_summary['per_exchange_data'][ex_id]
            ex_summary['sum_price_vol'] += price * volume
            ex_summary['total_volume'] += volume
            if trade.get('is_buy'):
                ex_summary['buy_volume'] += volume

        return interval_summary

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

            # Normalize reward
            if self.previous_marked_to_market > 0:
                reward = reward / self.previous_marked_to_market

            self.previous_marked_to_market = marked_to_market

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
            pnl = marked_to_market - self.starting_cash
            if pnl < 0:
                scaled_loss = pnl  / self.starting_cash
                reward = - (scaled_loss ** 2)
            else:
                reward = pnl / self.starting_cash
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

        true_m2m_value = self._calculate_true_m2m(raw_state)
        market_price = mkt_data[0].get("last_transaction", 0) if mkt_data else 0



        # --- Global Agent Information ---
        info = {
            "true_marked_to_market": true_m2m_value,
            "current_time": internal_data.get("current_time"),
            "cash": internal_data.get("cash"),
            "market_price": market_price,
            "realised_pnl": self.realised_pnl,
            "holdings_by_exchange": internal_data.get("holdings_by_exchange", 0),
            "withdrawal_fees": internal_data.get("withdrawal_fees", {}),
            "total_holdings": internal_data.get("total_holdings", 0),
            "order_status": internal_data.get("order_status"),
            # Pass the processed state to the calculation function
            "mkt_open_times": internal_data.get("mkt_open_times"),
            "mkt_close_times": internal_data.get("mkt_close_times"),
            # This adds the mask to the info dict that gets passed to the agent
            "action_mask": self._get_action_mask()
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
        return info