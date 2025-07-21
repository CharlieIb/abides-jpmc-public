# cdormsc02 (Cryptocurrency Data Oracle Reference Market Simulation Configuration 02)
# - X     Data Oracle
# - 2     Exchange Agent
# - 3     Adaptive Market Maker Agents
# - 200   Value Agents
# - 35    Momentum Agents
# - 20000  Noise Agents

"""
Character -
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    ExchangeAgent
)
from abides_multi_exchange.agents import (
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
    ArbitrageAgent
)
from abides_markets.models import OrderSizeModel
from oracle import DataOracle
from abides_markets.utils import generate_latency_model


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
        # --- General Simulation Parameters ---
    seed=int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1),
    date="20250611",
    mkt_open_time="00:10:00",
    end_time="23:59:59",
    stdout_log_level="INFO",
    ticker="ABM",
    starting_cash=1_000_000,  # Cash in this simulator is always in CENTS.

    # --- Log orders: Individual agent logs ---
    # These must be None or True (False will yield True)
    log_orders_value=True,
    log_orders_momentum=True,
    log_orders_arbitrage=True,
    log_orders_MM=True,
    log_orders_noise=None,
    log_orders=False,

    # --- Exchange Agent Parameters ---
    num_exchange_agents=2,
    book_logging=True,
    book_log_depth=10,
    stream_history_length=500,
    exchange_log_orders=True, # overall market logs and file creation

    # --- Data Oracle Parameters ---
    data_file_path="/home/charlie/PycharmProjects/ABIDES_GYM_EXT/abides-jpmc-public/my_experiments/gym_crypto_markets/data/test/BTCUSDT-trades-2025-06-11-1s.csv",

    # --- Withdrawal Fee Parameters ---
    withdrawal_fees_enabled=True,

    # --- Population Parameters ---
    num_value_agents=100,
    num_momentum_agents=35,
    num_arbitrage_agents=5,
    # num_mm_agents = defined below
    num_noise_agents=2000,

    # --- Value Agent Parameters ---
    value_kappa=1.67e-15, # appraisal of mean reversion
    value_lambda_a=5.7e-12,  # arrival rate

    # --- Momentum Agent Parameters ---
    momentum_min_size=1,
    momentum_max_size=10,
    momentum_poisson_arrival=True,
    momentum_wake_up_freq="37s", # TODO: Consider changing this to a poisson distribution
    momentum_subscribe=False,   # Explicitly set to polling mode


    # --- Arbitrage Agents
    arbitrage_wake_up_freq="60s",
    arbitrage_min_profit_margin=1,
    arbitrage_pov=0.01,
    arbitrage_max_inventory=100,

    # --- Market Maker Agents ---
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_wake_up_freq="60s",
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=20, # Doubled from baseline, as BTCUSDT spreads are wider than stocks due ot higher volatility
    mm_min_order_size=1,
    mm_skew_beta=0.1, # Was zero and tends to impact inventory risk aversion (don't kno why this was zero)
    mm_price_skew=6, # response to momentum agents, able to shift response to momentum in market
    mm_level_spacing=7, # Increase to create less dense OB with larger gaps between price levels  as you would expect to see in crypto
    mm_spread_alpha=0.85, # Increased from baseline of 0.75, BTC tends to be more sensitive to volatility widening spreads
    mm_backstop_quantity=0,
    mm_cancel_limit_delay=50,  # 50 nanoseconds
):
    # fix seed
    np.random.seed(seed)


    def path_wrapper(pomegranate_model_json):
        """
        temporary solution to manage calls from abides-gym or from the rest of the code base
        TODO:find more general solution
        :return:
        :rtype:
        """
        # get the  path of the file
        path = os.getcwd()
        if path.split("/")[-1] == "abides_gym":
            return "../" + pomegranate_model_json
        else:
            return pomegranate_model_json

    # --- Date and Time ---
    DATE = int(pd.to_datetime(date).to_datetime64())
    MKT_OPEN = DATE + str_to_ns(f"{mkt_open_time}")
    MKT_CLOSE = DATE +str_to_ns(f"{end_time}")

    # These times needed for distribution of arrival times of Noise Agents
    NOISE_MKT_OPEN = DATE + str_to_ns(f"{mkt_open_time}")
    NOISE_MKT_CLOSE = DATE + str_to_ns("23:00:00")


    # ---- MM PARAMS -------
    MM_PARAMS = [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
        # You could add more tuples here for more MMs with different strategies
    ]
    num_mm_agents = len(MM_PARAMS)

    # Agent configuration
    agent_count, agents, agent_types = 0, [], []

    #
    exchange_agents = [
        ExchangeAgent(
            id=j,
            name=f"EXCHANGE_AGENT_{j}",  # Use f-string for clarity
            type="ExchangeAgent",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            symbols=[ticker],
            book_logging=book_logging,
            book_log_depth=book_log_depth,
            log_orders=exchange_log_orders,
            pipeline_delay=0,
            computation_delay=0,
            stream_history=stream_history_length,
            random_state=np.random.RandomState(
                seed=np.random.randint(low=0, high=2**32, dtype="uint64")
            ),
            )
            for j in range(agent_count, agent_count + num_exchange_agents)
    ]
    exchange_ids = [agent.id for agent in exchange_agents]
    agents.extend(exchange_agents)
    agent_count += num_exchange_agents
    agent_types.extend(["ExchangeAgent"])

    # Dynamically create the fee structure based on the number of exchanges.
    withdrawal_fees = {}
    if withdrawal_fees_enabled:
        for ex_id in exchange_ids:
            # Example: fee increases with exchange ID
            fee = 5 + (ex_id * 2)
            withdrawal_fees[ex_id] = {'default': fee, ticker: fee}


    # Oracle
    # This setup uses a single data source, meaning all exchanges share the same
    # fundamental price series. Arbitrage will come from temporary imbalances.
    print(" --- Dynamically configuring simulation from daily data ---")
    try:
        df = pd.read_csv(data_file_path)
        daily_mean_price = df['PRICE'].mean()
        daily_volatility = df['PRICE'].std()
        print(f"Data for {date}: Mean Price (cents) = {daily_mean_price:.2f}, Volatility = {daily_volatility:.2f}")
    except FileNotFoundError:
        print(f"Warning: Data file not found at {data_file_path}. Using default parameters.")
        daily_mean_price = 1000
        daily_volatility = 50

    symbols = {ticker: {'data_file': data_file_path}}
    oracle = DataOracle(MKT_OPEN, MKT_CLOSE, symbols)

    r_bar = int(daily_mean_price)
    sigma_n = daily_volatility * 0.05
    kappa = value_kappa
    lambda_a = value_lambda_a
    ORDER_SIZE_MODEL = OrderSizeModel()


    # Value Agents
    agents.extend([
        ValueAgent(
            id=j, name=f"Value Agent {j}", type="ValueAgent", symbol=ticker,
            starting_cash=starting_cash, sigma_n=sigma_n, r_bar=r_bar, kappa=kappa,
            lambda_a=lambda_a, log_orders=log_orders_value, order_size_model=ORDER_SIZE_MODEL,
            exchange_ids=exchange_ids,  # Connect to all exchanges
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        )
        for j in range(agent_count, agent_count + num_value_agents)
    ])
    agent_count += num_value_agents
    agent_types.extend(["ValueAgent"])

    # Momentum Agents
    agents.extend([
        MomentumAgent(
            id=j, name=f"MOMENTUM_AGENT_{j}", type="MomentumAgent", symbol=ticker,
            starting_cash=starting_cash, min_size=momentum_min_size, max_size=momentum_max_size,
            poisson_arrival=momentum_poisson_arrival, wake_up_freq=str_to_ns(momentum_wake_up_freq),
            log_orders=log_orders_momentum, order_size_model=ORDER_SIZE_MODEL,
            subscribe=momentum_subscribe,
            exchange_ids=exchange_ids,  # Connect to all exchanges
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        )
        for j in range(agent_count, agent_count + num_momentum_agents)
    ])
    agent_count += num_momentum_agents
    agent_types.extend(["MomentumAgent"])

    # Arbitrage Agents
    agents.extend([
        ArbitrageAgent(
            id=j, name=f"Arbitrage Agent {j}", type="ArbitrageAgent", symbol=ticker,
            starting_cash=starting_cash, wake_up_freq=str_to_ns(arbitrage_wake_up_freq),
            pov=arbitrage_pov, max_inventory=arbitrage_max_inventory, min_profit_margin=arbitrage_min_profit_margin,
            log_orders=log_orders_arbitrage,
            exchange_ids=exchange_ids,  # Must connect to all exchanges
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        )
        for j in range(agent_count, agent_count + num_arbitrage_agents)
    ])
    agent_count += num_arbitrage_agents
    agent_types.extend(["ArbitrageAgent"])

    for i in range(num_mm_agents):
        exchange_to_assign = exchange_ids[i % len(exchange_ids)]
        agents.append(
            AdaptiveMarketMakerAgent(
                id=agent_count + i, name=f"ADAPTIVE_MM_{i}_EX_{exchange_to_assign}",
                type="AdaptivePOVMarketMakerAgent", symbol=ticker, starting_cash=starting_cash,
                exchange_id=exchange_to_assign,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
                log_orders=log_orders_MM,  poisson_arrival=True,window_size=MM_PARAMS[i][0],
                pov=MM_PARAMS[i][1], num_ticks=MM_PARAMS[i][2], wake_up_freq=MM_PARAMS[i][3],
                min_order_size=MM_PARAMS[i][4], skew_beta=mm_skew_beta, price_skew_param=mm_price_skew,
                level_spacing=mm_level_spacing, spread_alpha=mm_spread_alpha,
                backstop_quantity=mm_backstop_quantity, cancel_limit_delay=mm_cancel_limit_delay,
            )
        )
    agent_count += num_mm_agents
    agent_types.extend(["POVMarketMakerAgent"])
    # Noise Agents
    agents.extend([
        NoiseAgent(
            id=j, name=f"NoiseAgent {j}", type="NoiseAgent", symbol=ticker,
            starting_cash=starting_cash, log_orders=log_orders_noise,
            wakeup_time=get_wake_time(NOISE_MKT_OPEN,NOISE_MKT_CLOSE),
            order_size_model=ORDER_SIZE_MODEL,
            exchange_ids=exchange_ids,  # Connect to all exchanges
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        )
        for j in range(agent_count, agent_count + num_noise_agents)
    ])
    agent_count += num_noise_agents
    agent_types.extend(["NoiseAgent"])

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
    # LATENCY
    latency_model = generate_latency_model(agent_count)
    default_computation_delay = 50  # 50 nanoseconds

    # Final kernel configuration
    kernelStartTime = DATE + str_to_ns("00:00:00")
    kernelStopTime = MKT_CLOSE + str_to_ns("1s")

    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "mkt_open": MKT_OPEN,
        "mkt_close": MKT_CLOSE,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle, "withdrawal_fees": withdrawal_fees},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
        "skip_log" : False,
        "num_exchange_agents" : num_exchange_agents,
    }
