# cdormsc01 (Cryptocurrency Data Oracle Reference Market Simulation Configuration 01)
# - X     Data Oracle
# - 1     Exchange Agent
# - 3     Adaptive Market Maker Agents
# - 200   Value Agents
# - 35    Momentum Agents
# - 20000  Noise Agents

"""
Character - This market is characterised by quite large swings and then stabilisation.
Sudden clusters of Noise agents create momentum which is piled on by momentum agents reducing the price by ~25%
Then stabilising forces of the MM and Values agents raise the price back to fundamental value
More prone to reducing then increasing, however surges did happen
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
)
from abides_markets.models import OrderSizeModel
from oracle import DataOracle
from abides_markets.utils import generate_latency_model


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    seed=int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1),
    date="20250611",
    end_time="23:59:59",
    stdout_log_level="INFO",
    ticker="ABM",
    starting_cash=1_000_000,  # Cash in this simulator is always in CENTS.
    # Individual agent logs - These must be None or True (False will yield True)
    log_orders_noise=None,
    log_orders_momentum=True,
    log_orders_MM=True,
    log_orders_value=True,
    log_orders=False,
    # 1) Exchange Agent
    book_logging=True,
    book_log_depth=10,
    stream_history_length=500,
    exchange_log_orders=True, # overall market logs and file creation?
    # 2) Noise Agent
    num_noise_agents=20000,
    # 3) Value Agents
    num_value_agents=200,
    r_bar=1_100,  # true mean fundamental value
    kappa=1.67e-15,  # Value Agents appraisal of mean-reversion
    lambda_a=5.7e-12,  # ValueAgent arrival rate
    # oracle - commented out as using data oracle
    # kappa_oracle=1.67e-16,  # Mean-reversion of fundamental time series.
    # sigma_s=0,
    # fund_vol=5e-5,  # Volatility of fundamental time series (std).
    # megashock_lambda_a=2.77778e-18,
    # megashock_mean=1000,
    # megashock_var=50_000,
    # 4) Market Maker Agents
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=20, # Doubled from baseline, as BTCUSDT spreads are wider than stocks due ot higher volatility
    mm_wake_up_freq="60S",
    mm_min_order_size=1,
    mm_skew_beta=0.1, # Was zero and tends to impact inventory risk aversion (don't kno why this was zero)
    mm_price_skew=6, # response to momentum agents, able to shift response to momentum in market
    mm_level_spacing=7, # Increase to create less dense OB with larger gaps between price levels  as you would expect to see in crypto
    mm_spread_alpha=0.85, # Increased from baseline of 0.75, BTC tends to be more sensitive to volatility widening spreads
    mm_backstop_quantity=0,
    mm_cancel_limit_delay=50,  # 50 nanoseconds
    # 5) Momentum Agents
    num_momentum_agents=35,
):
    """
    create the background configuration for rmsc04
    These are all the non-learning agent that will run in the simulation
    :param seed: seed of the experiment
    :type seed: int
    :param log_orders: debug mode to print more
    :return: all agents of the config
    :rtype: list
    """

    # fix seed
    np.random.seed(seed)

    # --- Pre-computation Block ---
    # Adapt r_bar and sigma_n dynamically for data

    print(" --- Dynamically configuring simulation from daily data ---")

    # Construct the file path based on the date parameter
    # Important: Assumes your scaled data is in this location

    data_file_path = f"/data/test/BTCUSDT-trades-2025-06-11-1s.csv"

    try:
        df = pd.read_csv(data_file_path)
        # Assumes the price column is named 'PRICE' and is already scaled to cents
        daily_mean_price = df['PRICE'].mean()
        daily_volatility = df['PRICE'].std()

        print(f"Data for {date}: Mean Price (cents) = {daily_mean_price:.2f}, Volatility = {daily_volatility:.2f}")

    except FileNotFoundError:
        print(f"Warning: Data file not found at {data_file_path}. Using default parameters.")
        daily_mean_price = 1000
        daily_volatility = 50

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

    mm_wake_up_freq = str_to_ns(mm_wake_up_freq)

    # order size model
    ORDER_SIZE_MODEL = OrderSizeModel()  # Order size model
    # market marker derived parameters
    MM_PARAMS = [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
    ]
    NUM_MM = len(MM_PARAMS)
    # noise derived parameters
    # Set r_bar dynamically based on data mean
    r_bar = int(daily_mean_price)
    #observation noise based on daily volatility
    SIGMA_N = daily_volatility * 0.05

    # date&time
    DATE = int(pd.to_datetime(date).to_datetime64())
    MKT_OPEN = DATE + str_to_ns("00:10:00")
    MKT_CLOSE = DATE + str_to_ns(end_time)
    # These times needed for distribution of arrival times of Noise Agents
    NOISE_MKT_OPEN = MKT_OPEN
    NOISE_MKT_CLOSE = DATE + str_to_ns("23:00:00")

    # Oracles
    # Sparse Mean Reverting Oracle
    '''
    symbols = {
        ticker: {
            "r_bar": r_bar,
            "kappa": kappa_oracle,
            "sigma_s": sigma_s,
            "fund_vol": fund_vol,
            "megashock_lambda_a": megashock_lambda_a,
            "megashock_mean": megashock_mean,
            "megashock_var": megashock_var,
            "random_state": np.random.RandomState(
                seed=np.random.randint(low=0, high=2**32)
            ),
        }
    }

    oracle = SparseMeanRevertingOracle(MKT_OPEN, NOISE_MKT_CLOSE, symbols)
    '''
    # Data Oracle

    symbols = {
        ticker: {
            'data_file': data_file_path,
        }
    }
    oracle = DataOracle(MKT_OPEN, NOISE_MKT_CLOSE, symbols)

    # Agent configuration
    agent_count, agents, agent_types = 0, [], []

    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
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
        ]
    )
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    agents.extend(
        [
            NoiseAgent(
                id=j,
                name="NoiseAgent {}".format(j),
                type="NoiseAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                wakeup_time=get_wake_time(NOISE_MKT_OPEN, NOISE_MKT_CLOSE),
                log_orders=log_orders_noise,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_noise_agents)
        ]
    )
    agent_count += num_noise_agents
    agent_types.extend(["NoiseAgent"])

    agents.extend(
        [
            ValueAgent(
                id=j,
                name="Value Agent {}".format(j),
                type="ValueAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                sigma_n=SIGMA_N,
                r_bar=r_bar,
                kappa=kappa,
                lambda_a=lambda_a,
                log_orders=log_orders_value,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_value_agents)
        ]
    )
    agent_count += num_value_agents
    agent_types.extend(["ValueAgent"])

    agents.extend(
        [
            AdaptiveMarketMakerAgent(
                id=j,
                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                type="AdaptivePOVMarketMakerAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                pov=MM_PARAMS[idx][1],
                min_order_size=MM_PARAMS[idx][4],
                window_size=MM_PARAMS[idx][0],
                num_ticks=MM_PARAMS[idx][2],
                wake_up_freq=MM_PARAMS[idx][3],
                poisson_arrival=True,
                cancel_limit_delay=mm_cancel_limit_delay,
                skew_beta=mm_skew_beta,
                price_skew_param=mm_price_skew,
                level_spacing=mm_level_spacing,
                spread_alpha=mm_spread_alpha,
                backstop_quantity=mm_backstop_quantity,
                log_orders=log_orders_MM,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for idx, j in enumerate(range(agent_count, agent_count + NUM_MM))
        ]
    )
    agent_count += NUM_MM
    agent_types.extend("POVMarketMakerAgent")

    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MOMENTUM_AGENT_{}".format(j),
                type="MomentumAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                min_size=1,
                max_size=10,
                wake_up_freq=str_to_ns("37s"),
                poisson_arrival=True,
                log_orders=log_orders_momentum,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_momentum_agents)
        ]
    )
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
    # LATENCY
    latency_model = generate_latency_model(agent_count)

    default_computation_delay = 50  # 50 nanoseconds

    ##kernel args
    kernelStartTime = DATE
    kernelStopTime = MKT_CLOSE + str_to_ns("1s")

    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
        "skip_log" : False
    }
