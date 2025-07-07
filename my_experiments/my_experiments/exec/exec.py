import gym
from tqdm import tqdm

import abides_gym
import gym_crypto_markets

from my_experiments.agents import MyRLAgent, MeanReversionAgent, DQNAgent


if __name__ == "__main__":

    env = gym.make(
        "CryptoEnv-v1",
        background_config="cdormsc01",
        timestep_duration='1s',
        debug_mode=True # Required for Mean reversion
    )

    env.seed(0)  # Set a seed for reproducibility

    # Instantiate your Mean Reversion Agent
    # You can adjust window and num_std_dev here
    agent = MeanReversionAgent(env.observation_space, env.action_space, window=20, num_std_dev=2)

    # Get the fixed order size and starting cash from the environment's configuration
    # This is vital for the agent's cash management logic and initial value.
    agent.order_fixed_size = env.unwrapped.order_fixed_size
    agent.initial_cash = env.unwrapped.starting_cash

    num_episodes = 1  # Define how many episodes to run for the simulation

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1} ---")
        # env.reset() now returns ONLY state for older Gym versions
        state = env.reset()
        agent.reset_agent_state()  # Reset the agent's internal state for the new episode

        # Initial cash is already set from env.unwrapped.starting_cash
        # We cannot get the initial price from info here, it will be available after the first step.

        done = False
        episode_reward = 0
        step_count = 0

        # Print the initial state for verification
        print(f"Initial State (Episode {episode + 1}):")
        print(f"  Holdings: {state[0][0]:.2f}")
        print(f"  Imbalance: {state[1][0]:.4f}")
        print(f"  Spread: {state[2][0]:.2f}")
        print(f"  Direction Feature: {state[3][0]:.2f}")
        print(f"  Price Changes (most recent first): {state[4]}")
        print(f"  Initial Cash (from env config): {agent.initial_cash:.2f}")

        info = {}

        # Use tqdm for a progress bar during the episode
        max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps else 10000  # Fallback if not defined
        with tqdm(total=max_steps, desc=f"Episode {episode + 1} Progress") as pbar:
            while not done and step_count < max_steps:
                # For the very first step, info will not contain 'last_transaction' or 'cash'
                # The agent's choose_action handles this by returning HOLD.
                # After the first step, info will be populated.
                action = agent.choose_action(state, info)  # Pass an empty dict for info initially

                if action != 1:
                    print(action)

                # 2. Environment takes a step with the chosen action
                # env.step returns (observation, reward, terminated, truncated, info)
                new_state, reward, done, info = env.step(action)  # Capture truncated and updated info
                # 3. Agent updates its internal state (no policy learning for rule-based)
                agent.update_policy(state, action, reward, new_state, done)  # Pass done or truncated

                # Update current state and episode reward
                state = new_state
                episode_reward += reward
                step_count += 1
                pbar.update(1)  # Update progress bar


        # Calculate final portfolio value using the last known cash and price from info
        # These should be available in the 'info' from the very last step
        final_cash = info.get("cash", 0.0)
        final_holdings = state[0][0]  # Holdings from the last state observation
        final_price = info.get("last_transaction", 0.0)  # Last price from the final info dict

        final_portfolio_value = final_cash + (final_holdings * final_price)

        print(f"Episode {episode + 1} finished after {step_count} steps.")
        print(
            f"  Total Reward (Normalized Change in Portfolio Value): {episode_reward:.4f}")  # Reward is normalized by order_fixed_size and time
        print(f"  Initial Portfolio Value: {agent.initial_cash:.2f}")
        print(f"  Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"  Raw P&L (Final - Initial): {(final_portfolio_value - agent.initial_cash):.2f}")
        print(f"  Final State: Cash={final_cash:.2f}, Holdings={final_holdings:.2f}, Price={final_price:.2f}")

    env.close()  # Close the environment
    print("\nSimulation finished.")