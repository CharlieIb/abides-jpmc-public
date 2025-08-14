import argparse
import yaml
import gym
from tqdm import tqdm
import importlib
import os
import csv
from datetime import datetime
import random
import numpy as np

# --- Local Imports ---
import gym_crypto_markets

from gym_crypto_markets.configs.cdormsc02 import build_config
from gym_crypto_markets.envs.hist_crypto_env_v02 import HistoricalTradingEnv_v02

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run an ABIDES-Gym simulation from a YAML config file.")
    parser.add_argument('config_path', type=str, help="Path to the YAML configuration file.")
    parser.add_argument('--mode', type=str, default='train-abides',
                        choices=['train-abides', 'train-historical', 'test-historical', 'train-historical-se', 'train-abides-se'],
                        help="The mode to run the simulation in.")
    parser.add_argument('--agent', type=str, default=None,
                        choices=['MeanReversionAgent', 'DQNAgent', 'PPOAgent', 'SETripleBarrier', 'METripleBarrier'],
                        help="Specify the agent configuration to use from the YAML file (e.g., SingleExchangeAgent).")

    parser.add_argument('--date', type=str, default=None,
                        help="Specify a date for this simulation to run (if it is available then it will be loaded)")

    parser.add_argument('--load_weights_path', type=str, default=None,
                        help="Path to pre-trained agent weights to load (for testing).")
    args = parser.parse_args()

    # Load parameters from YAML
    with open(args.config_path, 'r') as file:
        params = yaml.safe_load(file)
        print(f"Loaded configuration from {args.config_path}")

    # Extract parameter sections
    bg_params = params.get('background_config_params', {})

    if args.mode == 'train-historical-se':
        print("--- Single-Exchange mode activated. Overriding configuration. ---")

        # 1. Force number of exchanges to 1
        bg_params['exchange_params']['num_exchange_agents'] = 1
        print(f" > num_exchange_agents set to: 1")

        # 2. Filter data templates to only use 'binance'
        if 'binance' in bg_params['historical_templates']:
            binance_template = bg_params['historical_templates']['binance']
            bg_params['historical_templates'] = {'binance': binance_template}
            print(f" > hisotrical_templates filtered to 'binance' only.")
        else:
                raise ValueError("Single-exchange mode requires a 'binance' tempate in the config")
    elif args.mode == 'train-abides-se':
        print(" --- Single-Exchange mode activated. Overriding configuration. ---")

        #1. force the number of exchanges to 1
        bg_params['exchange_params']['num_exchange_agents'] = 1
        print(f" > num_exchange_agents set to: 1")
        bg_params['agent_populations']['num_arbitrage_agents'] = 0
        print(f" > num_arbitrage_agents set to: 1")

    env_params = params.get('gym_environment', {})
    runner_params = params.get('simulation_runner', {})
    if args.agent:
        active_agent_name = args.agent
        print(f"Using agent specified by command line: --agent {active_agent_name}")
    else:
        # Fall back to the YAML file if --agent is not provided
        active_agent_name = params.get('active_agent_config')
        print(f"Using default agent from YAML config: {active_agent_name}")

    if not active_agent_name:
        raise ValueError("No agent specified. Provide --agent or set 'active_agent_config' in the YAML file.")

    agent_params = params.get('agent_configurations', {}).get(active_agent_name)
    if not agent_params:
        raise ValueError(f"Could not find configuration for agent '{active_agent_name}' in the YAML file.")

    save_params = runner_params.get('save_weights', {})
    save_enabled = save_params.get('enabled', False)
    save_dir = save_params.get('directory', 'weights')
    save_freq_episodes = save_params.get('frequency_episodes', 1)
    if save_enabled and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    use_confidence_sizing = agent_params.get('use_confidence_sizing', False)
    env_params["use_confidence_sizing"] = use_confidence_sizing

    log_params = runner_params.get('performance_log', {})
    log_enabled = log_params.get('enabled', True)
    log_dir = log_params.get('directory', 'logs')
    log_freq_steps = log_params.get('frequency_steps', 1)
    if log_enabled and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = None
    csv_writer = None
    csv_file = None
    if log_enabled:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"{active_agent_name}_{args.mode}_log_{timestamp_str}.csv")
        print(f"Performance logging enabled. Saving to: {log_file_path}")

        # Open the file and create the writer
        csv_file = open(log_file_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        # Write the header row
        header = [
            'timestamp', 'episode', 'step', 'total_pnl', 'realised_pnl', 'reward',
            'portfolio_value', 'total_holdings', 'cash', 'action', 'market_price'
        ]
        csv_writer.writerow(header)

    summary_log_path = None
    summary_csv_writer = None
    summary_csv_file = None

    if log_enabled:
        summary_log_path = os.path.join(log_dir, f"{active_agent_name}_{args.mode}_episode_summary.csv")
        print(f"Episode summary logging enabled. Saving to: {summary_log_path}")

        summary_csv_file = open(summary_log_path, 'w', newline='')
        summary_csv_writer = csv.writer(summary_csv_file)

        summary_header = [
            'episode', 'total_pnl', 'sharpe_ratio', 'win_rate',
            'total_trades_closed', 'profit_exits', 'loss_exits', 'time_exits'
        ]
        summary_csv_writer.writerow(summary_header)

    # Initialise Environment
    # Build the background config dictionary that abides-gym needs
    if args.mode in ['train-abides', 'train-abides-se']:
        print("Initializing ABIDES simulation environment...")
        # Get the environment ID from the config
        env_id = env_params.pop('env_id', 'CryptoEnv-v2')
        # Create the Gym environment, passing the ABIDES config and other env params
        env = gym.make(env_id, background_config=bg_params, **env_params)
    else:  # train-historical, train-historical-se or test-historical
        print("Initializing Historical backtesting environment...")
        # We need to know the shape of the observation space and action space
        # For simplicity, we can hardcode them or create a dummy ABIDES env to get them
        # This assumes your state vector has 17 features and 7 actions

        env = HistoricalTradingEnv_v02(
            bg_params=bg_params,
            env_params=env_params,
        )

    env.seed(bg_params.get('seed', 0))
    num_exchanges = env.unwrapped.num_exchanges

    # Agent property imports
    agent_module_path = agent_params.pop('agent_module', None)
    agent_class_name = agent_params.pop('agent_class', None)

    if not agent_module_path or not agent_class_name:
        raise ValueError("Config file must specify 'agent_module' and 'agent_class' under agent_configurations'")

    # Dynamically import the module and get the class
    print(f"Attempting to import agent: {agent_class_name} from {agent_module_path}")
    agent_module = importlib.import_module(agent_module_path)
    AgentClass = getattr(agent_module, agent_class_name)

    agent = AgentClass(
        env.observation_space,
        env.action_space,
        # num_exchanges=num_exchanges, # Only needed with mean reversion agent
        **agent_params
    )
    # Configure agent with env-specific details
    agent.order_fixed_size = env.unwrapped.order_fixed_size
    agent.initial_cash = env.unwrapped.starting_cash

    # If testing, turn of exploration rate and load pre-trained weights
    if args.mode == 'test-historical':
        if not args.load_weights_path:
            raise ValueError("Must provide --load_weights_path in test mode.")
        agent.load_weights(args.load_weights_path)
        if hasattr(agent, 'exploration_rate'):
            agent.exploration_rate = 0.0

    # Simulation Loop
    num_episodes = runner_params.get('num_episodes', 1)
    max_steps = runner_params.get('max_episode_steps', 10000)


    # DATA PATH HANDLING
    specific_date = None

    historical_dates = bg_params.get('historical_dates')
    if args.date in historical_dates:
        specific_date = args.date
        print(f"\n Date {specific_date} is available, loading data into simulation .... \n")
    elif historical_dates:
        random.shuffle(historical_dates)
        print("Shuffled historical dates.")

    historical_templates = bg_params.get('historical_templates')
    hist_path_template_1 = historical_templates.get('binance')
    hist_path_template_2 = historical_templates.get('kraken')
    abides_path_template = bg_params.get('data_path_template')


    for episode in range(num_episodes):
        # Reset the environment and the agent's internal state for a new episode.
        # For older Gym versions (like 0.18.0), 'env.reset()' returns only the initial state.

        # Select a new data path for this episode, cycling through the shuffled
        current_date = historical_dates[episode % len(historical_dates)] if not specific_date else specific_date

        specific_date = None # Comment this out if you want the episode to replay on the same date

        if args.mode == 'train-abides':
            current_data_path = abides_path_template.format(current_date, current_date)

            print(f"\n--- Starting Episode {episode + 1}/{num_episodes} using data: {os.path.basename(current_data_path)} ---")

            # Create the override dictionary to pass to the reset method
            override_params = {'data_file_path': current_data_path}
            state = env.reset(override_bg_params=override_params)
        elif args.mode in ['train-historical', 'test-historical']:

            # Dynamically generate the paths for this episode using the templates
            current_data_path_1 = hist_path_template_1.format(current_date, current_date)
            current_data_path_2 = hist_path_template_2.format(current_date, current_date)
            print(current_data_path_1)
            print(current_data_path_2)

            print(f"\n--- Starting Episode {episode + 1}/{num_episodes} using data from date: {current_date} ---")

            # This is a crucial change. You need to override the data paths
            # for your environment for each episode.
            override_params = {
                'data_paths': [current_data_path_1, current_data_path_2]
            }
            state = env.reset(override_bg_params=override_params)
        else:
            current_date = historical_dates[episode % len(historical_dates)]

            # Dynamically generate the paths for this episode using the templates
            current_data_path_1 = hist_path_template_1.format(current_date, current_date)
            print(current_data_path_1)

            print(f"\n--- Starting Episode {episode + 1}/{num_episodes} using data from date: {current_date} ---")

            # This is a crucial change. You need to override the data paths
            # for your environment for each episode.
            override_params = {
                'data_paths': [current_data_path_1]
            }
            state = env.reset(override_bg_params=override_params)

        if args.mode != 'test-historical':
            agent.reset_agent_state()
        # Initialised episode-specific tracking variables.
        done = False
        episode_reward = 0
        step_count = 0
        info = {}

        step_returns = []
        previous_portfolio_value = env.unwrapped.starting_cash

        #  Initial State verification
        print(f"Initial State (Episode {episode + 1}):")
        print(f"  Global VWAP: {state[0][0]:.2f}")
        print(f"  Total traded volume: {state[1][0]:.2f}")
        print(f"  Total TVI: {state[2][0]:.2f}")
        print(f"  Volatility: {state[3][0]:.2f}")
        print(f"  Cash : {state[4][0]:.2f}")
        print(f"  Total Holdings: {state[5][0]:.2f}")
        print(f"  Pnl : {state[6][0]:.2f}")


        for ex_id in range(num_exchanges):
            start_index = 7 + (ex_id * 3)
            print(f"  Exchange {ex_id} VWAP difference: {state[start_index][0]:.4f}")
            print(f"  Exchange {ex_id} TVI: {state[start_index + 1][0]:.2f}")
            print(f"  Exchange {ex_id} Volatility: {state[start_index + 2][0]:.2f}")
        # The returns are now at the end of the state vector
        returns_start_index = 1 + (num_exchanges * 3)
        print(f"  Historical Global Returns: {state[returns_start_index:].flatten()}")
        print(f"  Initial Cash (from env config): {agent.initial_cash/100:.2f}")

        with tqdm(total=max_steps, desc=f"Episode {episode + 1}") as pbar:
            while (not done and step_count < max_steps if max_steps else not done):
                action_output = agent.choose_action(state, info)
                new_state, reward, done, info = env.step(action_output)

                if args.mode.startswith('train'):
                    if active_agent_name == "PPOAgent":
                        agent.update_policy(state, action_output, reward, new_state, done)
                        learn_interval = agent_params.get('learn_interval', 2048)
                        if (step_count + 1) % learn_interval == 0 or done:
                            print(f"\n--- Triggering PPO learn step at step {step_count} ---")
                            agent.learn()
                    else:  # For DQN and other off-policy agents
                        agent.update_policy(state, action_output, reward, new_state, done)



                state = new_state
                episode_reward += reward
                step_count += 1
                pbar.update(1)

                if log_enabled and (step_count % log_freq_steps == 0 or done):
                    portfolio_value = info.get("true_marked_to_market", 0.0)

                    # For the summary log, to calc Sharped ratio
                    if previous_portfolio_value > 0:  # Avoid division by zero
                        # Calculate the percentage return for this step
                        return_for_step = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
                        step_returns.append(return_for_step)

                    previous_portfolio_value = portfolio_value
                    # Update for the next step
                    total_pnl = portfolio_value - agent.initial_cash

                    realised_pnl = info.get('realised_pnl', 0.0)

                    total_holdings = info.get("total_holdings", 0)
                    cash = info.get("cash", 0.0)
                    market_price = info.get("market_price", 0.0)  # Also requires env modification
                    log_data = [
                        datetime.now().isoformat(),
                        episode + 1,
                        step_count,
                        f"{total_pnl / 100:.2f}",
                        f"{realised_pnl / 100:.2f}",
                        f"{reward:.6f}",
                        f"{portfolio_value / 100:.2f}",
                        total_holdings,
                        f"{cash / 100:.2f}",
                        str(action_output),
                        f"{market_price / 100:.2f}"
                    ]
                    csv_writer.writerow(log_data)
                    csv_file.flush()

                # Update progress bar variables
                if args.mode == 'train-abides':
                    market_data = info.get("market_data", {})
                    price_0 = market_data.get(0, {}).get("last_transaction", 0.0)
                    price_1 = market_data.get(1, {}).get("last_transaction", 0.0)
                    pbar.set_postfix(Exch0=f'{price_0 / 100:,.2f}', Exch1=f'{price_1 / 100:,.2f}',
                                     Holdings=info.get("total_holdings", 0))
                else:  # Historical mode
                    pbar.set_postfix(Value=f'{info.get("portfolio_value", 0) / 100:,.2f}',
                                     Holdings=info.get("total_holdings", 0))


        #  Episode Summary
        if args.mode == 'train-abides':
            final_val = info.get("true_marked_to_market", 0.0)
        else:
            final_val = info.get("true_marked_to_market", 0.0)
        initial_cash = env.unwrapped.starting_cash
        final_pnl = (final_val - initial_cash)/100
        print(f"\n--- Episode {episode + 1} Summary ---")
        print(f"  Final Portfolio Value: ${final_val/100:,.2f}")
        print(f"  Total P&L: ${final_pnl:,.2f}")

        if log_enabled:
            # Get diagnostics from the agent
            diagnostics = agent.get_episode_diagnostics()

            sharpe_ratio = 0.0

            if len(step_returns) > 1 and np.std(step_returns) > 0:
                # Calculate the raw Sharpe ratio for the period
                # For high-frequency steps, the risk-free rate is assumed to be 0
                mean_return = np.mean(step_returns)
                std_dev_return = np.std(step_returns)

                # Annualize it. For crypto (24/7) with 1-minute steps:
                # 60 mins/hr * 24 hrs/day * 365 days/year
                annualization_factor = np.sqrt(60 * 24 * 365)

                sharpe_ratio = (mean_return / std_dev_return) * annualization_factor

            summary_data = [
                episode + 1,
                f"{final_pnl:.2f}",
                f"{sharpe_ratio:.4f}",
                f"{diagnostics['win_rate']:.2%}",  # Format as percentage
                diagnostics['total_trades_closed'],
                diagnostics['profit_exits'],
                diagnostics['loss_exits'],
                diagnostics['time_exits']
            ]
            summary_csv_writer.writerow(summary_data)
            summary_csv_file.flush()

        # Save weights at the end of the episode if enabled
        if save_enabled and args.mode.startswith('train') and (episode + 1) % save_freq_episodes == 0:
            file_path = os.path.join(save_dir, f"{active_agent_name}_episode_{episode+1}.pth")
            agent.save_weights(file_path)


    #  Cleanup
    if csv_file:
        csv_file.close()
    if summary_csv_file:
        summary_csv_file.close()
    env.close()
    print("\n--------------------------Simulation finished--------------------------")