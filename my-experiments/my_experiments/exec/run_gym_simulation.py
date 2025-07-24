import argparse
import yaml
import gym
from tqdm import tqdm

# --- Local Imports ---
# Make sure your custom environments and agents are importable
import gym_crypto_markets

from my_experiments.agents_multi import MeanReversionAgent  # Import your RL agent
from gym_crypto_markets.configs.cdormsc02 import build_config  # Import your ABIDES config builder

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run an ABIDES-Gym simulation from a YAML config file.")
    parser.add_argument('config_path', type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load my parameters from YAML
    with open(args.config_path, 'r') as file:
        params = yaml.safe_load(file)
        print(f"Loaded configuration from {args.config_path}")

    # Extract parameter sections
    bg_params = params.get('background_config_params', {})
    env_params = params.get('gym_environment', {})
    agent_params = params.get('rl_agent', {})
    runner_params = params.get('simulation_runner', {})

    # Initialise Environment
    # Build the background config dictionary that abides-gym needs
    abides_bg_config = build_config(bg_params)

    # Get the environment ID from the config
    env_id = env_params.pop('env_id', 'CryptoEnv-v2')

    # Create the Gym environment, passing the ABIDES config and other env params
    env = gym.make(
        env_id,
        background_config=abides_bg_config,
        **env_params
    )
    env.seed(bg_params.get('seed', 0))
    num_exchanges = env.unwrapped.num_exchanges
    # Initialise Agent
    # consider making this dynamic
    agent = MeanReversionAgent(
        env.observation_space,
        env.action_space,
        num_exchanges=num_exchanges,
        **agent_params
    )
    # Configure agent with env-specific details
    agent.order_fixed_size = env.unwrapped.order_fixed_size
    agent.initial_cash = env.unwrapped.starting_cash


    # --- 5. Simulation Loop ---
    num_episodes = runner_params.get('num_episodes', 3)
    max_steps = runner_params.get('max_episode_steps', 1000)

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1}/{num_episodes} ---")
        # Reset the environment and the agent's internal state for a new episode.
        # For older Gym versions (like 0.18.0), 'env.reset()' returns only the initial state.
        state = env.reset()
        agent.reset_agent_state()

        # Initialised episode-specific tracking variables.
        done = False
        episode_reward = 0
        step_count = 0
        info = {}

        # --- Initial State verification ---
        print(f"Initial State (Episode {episode + 1}):")
        print(f"  Total Holdings: {state[0][0]:.2f}")
        for ex_id in range(num_exchanges):
            start_index = 1 + (ex_id * 3)
            print(f"  Exchange {ex_id} Imbalance: {state[start_index][0]:.4f}")
            print(f"  Exchange {ex_id} Spread: {state[start_index + 1][0]:.2f}")
            print(f"  Exchange {ex_id} Direction: {state[start_index + 2][0]:.2f}")
        # The returns are now at the end of the state vector
        returns_start_index = 1 + (num_exchanges * 3)
        print(f"  Historical Global Returns: {state[returns_start_index:].flatten()}")
        print(f"  Initial Cash (from env config): {agent.initial_cash/100:.2f}")

        with tqdm(total=max_steps, desc=f"Episode {episode + 1}") as pbar:
            while not done and step_count < max_steps:
                action = agent.choose_action(state, info)
                new_state, reward, done, info = env.step(action)

                # Update progress bar variables
                market_data = info.get("market_data", {})
                market_price_0 = market_data.get(0, {}).get("last_transaction", 0.0)
                market_price_1 = market_data.get(1, {}).get("last_transaction", 0.0)
                current_holdings = int(info.get("total_holdings", 0))
                cash = int(info.get("cash", 0))
                pbar.set_postfix(Exchange_0=f'{market_price_0/100:,.2f}',Exchange_1=f'{market_price_1/100:,.2f}', Holdings=current_holdings, Cash =f"${cash/100}")

                agent.update_policy(state, action, reward, new_state, done)

                state = new_state
                episode_reward += reward
                step_count += 1
                pbar.update(1)


        # --- Episode Summary ---
        final_val = info.get("true_marked_to_market", 0.0) / 100
        initial_cash = env.unwrapped.starting_cash / 100
        print(f"--- Episode {episode + 1} Summary ---")
        print(f"  Final Portfolio Value: ${final_val:,.2f}")
        print(f"  Total P&L: ${final_val - initial_cash:,.2f}")

    # --- Cleanup ---
    env.close()
    print("\n--------------------------Simulation finished--------------------------")