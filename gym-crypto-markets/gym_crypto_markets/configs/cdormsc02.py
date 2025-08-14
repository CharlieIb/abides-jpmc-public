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
from time import strftime
from typing import Dict
import re
import numpy as np
import pandas as pd

from abides_core.utils import get_wake_time, str_to_ns

from abides_multi_exchange.agents import (
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
    ArbitrageAgent,
    ExchangeAgent
)
from abides_multi_exchange.agents_gym import FinancialGymAgent
from ..models import OrderSizeModelSimple, OrderSizeModelNoise
from ..oracle import DataOracle
from abides_markets.utils import generate_latency_model


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(params: Dict):
    # fix seed
    seed = int(datetime.now().timestamp() * 1_000_000) % (2 ** 32 - 1)
    np.random.seed(seed)

    # Extract Parameters from the Dictionary ---
    # First sort out date
    data_file_path = params['data_file_path']
    date_str = None

    # If this is a 'reset', then we need to check the file path and determine date
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', data_file_path)
    if date_match:
        date_str = date_match.group(1)
        print(f"Successfully extracted date '{date_str}' from file path.")

    # If initial set upt, check if 'date' was passed in params as a fallback
    elif 'date' in params:
        date_str = params['date']
        print(f"Could not find date in file path, using provided date: '{date_str}'")

    # Check we have a date str.
    if not date_str:
        raise ValueError("Fatal: Simulation date could not be determined from data_file_path or a 'date' parameter.")


    mkt_open_time = params['mkt_open_time']
    end_time = params['end_time']
    ticker = params['ticker']
    starting_cash = params['starting_cash']
    model_type = params.get('order_size_model_type', 'simple')

    log_order_params = params['log_order_params']
    log_orders_value = log_order_params['log_orders_value']
    log_orders_momentum = log_order_params['log_orders_momentum']
    log_orders_arbitrage = log_order_params['log_orders_arbitrage']
    log_orders_MM = log_order_params['log_orders_MM']
    log_orders_noise = log_order_params['log_orders_noise']
    log_orders = log_order_params['log_orders']

    # Exchange Agent parameters
    exchange_params = params['exchange_params']
    num_exchange_agents = exchange_params['num_exchange_agents']

    # Agent Populations
    agent_populations = params['agent_populations']
    num_value_agents = agent_populations['num_value_agents']
    num_momentum_agents = agent_populations['num_momentum_agents']
    num_arbitrage_agents = agent_populations['num_arbitrage_agents']
    num_noise_agents = agent_populations['num_noise_agents']

    # Agent Parameters
    value_params = params['value_params']
    momentum_params = params['momentum_params']
    arbitrage_params = params['arbitrage_params']
    mm_params = params['mm_params']


    # Set the seed for reproducibility
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
    DATE = int(pd.to_datetime(date_str).to_datetime64())
    MKT_OPEN = DATE + str_to_ns(f"{mkt_open_time}")
    MKT_CLOSE = DATE +str_to_ns(f"{end_time}")

    # These times needed for distribution of arrival times of Noise Agents
    NOISE_MKT_OPEN = DATE + str_to_ns(f"{mkt_open_time}")
    NOISE_MKT_CLOSE = DATE + str_to_ns("23:00:00")


    # ---- MM PARAMS -------
    MM_PARAMS = [
        (mm_params['window_size'], mm_params['pov'], mm_params['num_ticks'], mm_params['wake_up_freq'], mm_params['min_order_size']),
        (mm_params['window_size'], mm_params['pov'], mm_params['num_ticks'], mm_params['wake_up_freq'], mm_params['min_order_size']),
        (mm_params['window_size'], mm_params['pov'], mm_params['num_ticks'], mm_params['wake_up_freq'], mm_params['min_order_size']),
        (mm_params['window_size'], mm_params['pov'], mm_params['num_ticks'], mm_params['wake_up_freq'], mm_params['min_order_size']),
    ]
    num_mm_agents = len(MM_PARAMS)

    # Oracle
    # This setup uses a single data source, meaning all exchanges share the same
    # fundamental price series. Arbitrage will come from temporary imbalances.
    print(" --- Dynamically configuring simulation from daily data ---")
    try:
        df = pd.read_csv(data_file_path)
        daily_mean_price = df['PRICE'].mean()
        daily_volatility = df['PRICE'].std()

        timestamp_seconds = DATE / 1_000_000_000.0
        date_object = datetime.fromtimestamp(timestamp_seconds)
        formatted_date = date_object.strftime("%Y%m%d")
        print(f"Data for {formatted_date}: Mean Price (cents) = {daily_mean_price:.2f}, Volatility = {daily_volatility:.2f}")
    except FileNotFoundError:
        print(f"Warning: Data file not found at {data_file_path}. Using default parameters.")
        daily_mean_price = 11_000_000
        daily_volatility = 5_000

    symbols = {ticker: {'data_file': data_file_path}}
    oracle = DataOracle(MKT_OPEN, MKT_CLOSE, symbols)

    r_bar = int(daily_mean_price)
    sigma_n = daily_volatility * 0.05
    kappa = value_params['kappa']
    lambda_a = value_params['lambda_a']

    if model_type == 'realistic':
        print("--- Using agent-specific 'Realistic' order size models. ---" )
        model = OrderSizeModelNoise
    else: # 'simple'
        print("--- Using single 'Simple' order size model for all agents. ---")
        model = OrderSizeModelSimple


    NOISE_ORDER_SIZE_MODEL = model('noise')
    VALUE_ORDER_SIZE_MODEL = model('value')
    MOMENTUM_ORDER_SIZE_MODEL = model('momentum')


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
            book_logging=exchange_params['book_logging'],
            book_log_depth=exchange_params['book_log_depth'],
            log_orders=exchange_params['exchange_log_orders'],
            pipeline_delay=0,
            computation_delay=0,
            stream_history=exchange_params['stream_history_length'],
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
    if params.get('withdrawal_fees_enabled', False):
        for ex_id in exchange_ids:
            fee = r_bar * 15
            # IMPORTANT: adjust this to your simulation parameter, then adjust respective agents
            # E.G arbitrage agent's strategy will likely highly depend on this.
            withdrawal_fees[ex_id] = {'default': fee, ticker: fee}

    # Value Agents
    agents.extend([
        ValueAgent(
            id=j, name=f"Value Agent {j}", type="ValueAgent", symbol=ticker,
            starting_cash=starting_cash, sigma_n=sigma_n, r_bar=r_bar, kappa=kappa,
            lambda_a=lambda_a, log_orders=log_orders_value, order_size_model=VALUE_ORDER_SIZE_MODEL,
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
            starting_cash=starting_cash, min_size=momentum_params['min_size'], max_size=momentum_params['max_size'],
            poisson_arrival=momentum_params['poisson_arrival'], wake_up_freq=str_to_ns(momentum_params['wake_up_freq']),
            log_orders=log_orders_momentum, order_size_model=MOMENTUM_ORDER_SIZE_MODEL,
            subscribe=momentum_params['subscribe'],
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
            starting_cash=starting_cash, wake_up_freq=str_to_ns(arbitrage_params['wake_up_freq']),
            pov=arbitrage_params['pov'], max_inventory=arbitrage_params['max_inventory'], min_profit_margin=arbitrage_params['min_profit_margin'],
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
                min_order_size=MM_PARAMS[i][4], skew_beta=mm_params['skew_beta'], price_skew_param=mm_params['price_skew'],
                level_spacing=mm_params['level_spacing'], spread_alpha=mm_params['spread_alpha'],
                backstop_quantity=mm_params['backstop_quantity'], cancel_limit_delay=mm_params['cancel_limit_delay'],
            )
        )
    agent_count += num_mm_agents
    agent_types.extend(["POVMarketMakerAgent"])

    agents.extend([
        FinancialGymAgent(
            id=agent_count,
            name="GYM_AGENT",
            type="FinancialGymAgent",
            symbol=ticker,
            starting_cash=starting_cash,
            log_orders=log_orders,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        )
    ])
    agent_count += 1
    agent_types.extend(["FinancialGymAgent"])

    # Noise Agents
    agents.extend([
        NoiseAgent(
            id=j, name=f"NoiseAgent {j}", type="NoiseAgent", symbol=ticker,
            starting_cash=starting_cash, log_orders=log_orders_noise,
            wakeup_time=get_wake_time(NOISE_MKT_OPEN,NOISE_MKT_CLOSE),
            order_size_model=NOISE_ORDER_SIZE_MODEL,
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
    default_computation_delay = params['default_computation_delay']  # 50 nanoseconds

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
        "stdout_log_level": params['stdout_log_level'],
        "skip_log" : False,
        "num_exchange_agents" : num_exchange_agents,
    }
