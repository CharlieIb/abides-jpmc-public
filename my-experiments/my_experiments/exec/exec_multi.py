import gym
from tqdm import tqdm

# --- Local Imports ---
# Import the ABIDES-Gym environment and custom agents.
# These imports also trigger the registration of custom Gym environments
import gym_crypto_markets

from my_experiments.agents_multi import MeanReversionAgent


if __name__ == "__main__":
    """
    Main execution block to run a trading simulation using a custom Gym environment
    modelling crypto currency markets
    
    This script initialises a custom crypto trading environment from the ABIDES-Gym 
    framework, sets up a rule-based MeanReversionAgent, and runs a simulation
    for a specified number of episodes. It demonstrates the standard workflow of
    an RL loop: reset, step, close.
    """

    # 1. --- Environment Initialisation ---
    # Instatiate the custom trading environment using its registered ID.
    # 'debug_mode'=True is required to populate the 'info' dictionary with
    #extra data like 'cash' and 'last_transaction', which the MeanReversionAgent needs.


    env = gym.make(
        "CryptoEnv-v2",
        background_config="cdormsc02",
        timestep_duration='1s',
        debug_mode=True, # Required for Mean reversion
        first_interval="00:05:00",
    )

    env.seed(0)  # Set a seed for reproducibility

    # 2. ---Agent Initialisation ---
    num_exchanges = env.unwrapped.num_exchanges
    agent = MeanReversionAgent(env.observation_space,
                               env.action_space,
                               window=300,
                               num_std_dev=2,
                               num_exchanges=num_exchanges
                               )

    # Configure the agent with critical parameters from the environment's config
    agent.order_fixed_size = env.unwrapped.order_fixed_size
    agent.initial_cash = env.unwrapped.starting_cash

    # --- Simulation loop ---
    num_episodes = 1

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1} ---")
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
        print(f"  Initial Cash (from env config): {agent.initial_cash:.2f}")


        # tqdm for progress bar visualisation
        # max_steps are from the environment if they are available
        max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps else 20000  # Fallback if not defined
        with tqdm(total=max_steps, desc=f"Episode {episode + 1} Progress") as pbar:
            while not done and step_count < max_steps:
                action = agent.choose_action(state, info)
                new_state, reward, done, info = env.step(action)

                # Update progress bar variables
                market_data = info.get("market_data", {})
                market_price_0 = market_data.get(0, {}).get("last_transaction", 0.0)
                market_price_1 = market_data.get(1, {}).get("last_transaction", 0.0)
                current_holdings = int(info.get("total_holdings", 0))
                pbar.set_postfix(Price_0=f'{market_price_0:,.2f}',Price_1=f'{market_price_1:,.2f}', Holdings=current_holdings)

                # --- Agent Updates ---
                agent.update_policy(state, action, reward, new_state, done)

                # Update current state and accumulate episode reward
                state = new_state
                episode_reward += reward
                step_count += 1
                pbar.update(1)

        # --- Episode Summary ---
        final_portfolio_value = info.get("true_marked_to_market", 0.0)
        final_cash = info.get("cash", 0.0)
        final_holdings = info.get("total_holdings", 0.0)

        print(f"\n--- EPISODE {episode + 1} SUMMARY ---")
        print(f"Episode finished after {step_count} steps.")
        print(f"  Initial Portfolio Value: {agent.initial_cash:,.2f}")
        print(f"  Final Portfolio Value (Fee-Aware): {final_portfolio_value:,.2f}")
        print(f"  Raw P&L (Final - Initial): {(final_portfolio_value - agent.initial_cash):,.2f}")
        print(f"  Final State: Cash={final_cash:,.2f}, Total Holdings={final_holdings:.2f}")

    # 6. --- Cleanup ---
    env.close()
    print("\n--------------------------Simulation finished--------------------------")