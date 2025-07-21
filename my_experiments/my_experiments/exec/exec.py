import gym
from tqdm import tqdm

# --- Local Imports ---
# Import the ABIDES-Gym environment and custom agents.
# These imports also trigger the registration of custom Gym environments
import abides_gym
import gym_crypto_markets

from my_experiments.agents import MyRLAgent, MeanReversionAgent, DQNAgent


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
        "CryptoEnv-v1",
        background_config="cdormsc01",
        timestep_duration='1s',
        debug_mode=True # Required for Mean reversion
    )

    env.seed(0)  # Set a seed for reproducibility

    # 2. ---Agent Initialisation ---
    # Instantiate the rule-based agent and pass it the environment's observation
    # and action spaces. Hyperparameters like the Bollinger Band window can be set here.
    # You can adjust window and num_std_dev here

    agent = MeanReversionAgent(env.observation_space, env.action_space, window=300, num_std_dev=2)

    # Configure the agent with critical parameters from the environment's config
    # This ensures the agent's internal logic matches the simulation's rules.
    agent.order_fixed_size = env.unwrapped.order_fixed_size
    agent.initial_cash = env.unwrapped.starting_cash

    # --- Simulation loop ---
    num_episodes = 1  # Define how many episodes to run

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
        print(f"  Holdings: {state[0][0]:.2f}")
        print(f"  Imbalance: {state[1][0]:.4f}")
        print(f"  Spread: {state[2][0]:.2f}")
        print(f"  Direction Feature: {state[3][0]:.2f}")
        print(f"  Price Changes (most recent first): {state[4]}")
        print(f"  Initial Cash (from env config): {agent.initial_cash:.2f}")


        # tqdm for progress bar visualisation
        # max_steps are from the environment if they are available
        max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps else 100  # Fallback if not defined
        with tqdm(total=max_steps, desc=f"Episode {episode + 1} Progress") as pbar:
            while not done and step_count < max_steps:
                # 3. --- Agent Chooses Action ---
                # The agent selects an action based on the current state and info.
                # On the first step, 'info' is empty; the agent is designed to handle this.
                action = agent.choose_action(state, info)

                # 4. --- Environment Takes a Step ---
                # The environment processes the action and returns the new state,
                # reward, done flag, and updated info dictionary
                new_state, reward, done, info = env.step(action)

                # Update progress bar variables
                current_price = info.get("last_transaction", 0.0)
                current_holdings = int(state[0][0])
                pbar.set_postfix(Price=f'{current_price:,.2f}', Holdings=current_holdings)

                # 5. --- Agent Updates --- (no policy for rule-based learning)
                # For a rule-based agent, this step might be used for logic or
                # updating internal trackers, but no learning occurs
                agent.update_policy(state, action, reward, new_state, done)

                # Update current state and accumulate episode reward
                state = new_state
                episode_reward += reward
                step_count += 1
                pbar.update(1)

        # --- Episode Summary ---
        final_cash = info.get("cash", 0.0)
        final_holdings = state[0][0]
        final_price = info.get("last_transaction", 0.0)

        final_portfolio_value = final_cash + (final_holdings * final_price)

        print(f"Episode {episode + 1} finished after {step_count} steps.\n")
        # print(f"  Total Reward (Normalized Change in Portfolio Value): {episode_reward:.4f}")
        print(f"\n---------------------EPISODE {episode + 1} SUMMARY----------------------\n")
        print(f"  Initial Portfolio Value: {agent.initial_cash:.2f}")
        print(f"  Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"  Raw P&L (Final - Initial): {(final_portfolio_value - agent.initial_cash):.2f}")
        print(f"  Final State: Cash={final_cash:.2f}, Holdings={final_holdings:.2f}, Price={final_price:.2f}")
        print(f"  Final Portfolio Value: {final_portfolio_value:.2f}\n")
    # 6. --- Cleanup ---
    env.close()
    print("\n--------------------------Simulation finished--------------------------")