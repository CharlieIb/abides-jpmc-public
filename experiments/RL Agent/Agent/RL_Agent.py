import gym
from tqdm import tqdm

# Import to register environments
import abides_gym


# --- Define your RL Agent Class ---
# This is a conceptual class. In a real scenario, you'd use a library
# like Stable Baselines3, RLlib (as in the paper), or implement a custom
# agent (e.g., Q-learning, Policy Gradient, DQN from scratch).
class MyRLAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        # Initialize your policy, Q-table, neural network, etc. here
        print(
            f"Agent initialized with obs space: {observation_space.shape} and action space: {action_space.n if hasattr(action_space, 'n') else action_space.shape}")

    def choose_action(self, state):
        # This is where your agent's "brain" decides what to do
        # Based on the current 'state', return an action.
        # For a discrete action space (like BUY/HOLD/SELL), this would be an integer.
        # For this example, let's just pick a random action as a placeholder:
        action = self.action_space.sample()
        return action

    def update_policy(self, old_state, action, reward, new_state, done):
        # This is where your agent learns from the experience.
        # Implement your learning algorithm (e.g., Q-learning update,
        # neural network training for DQN/PPO, etc.)
        # For demonstration, we'll just print:
        # print(f"Agent learning: old_state={old_state[:2]}..., action={action}, reward={reward}, new_state={new_state[:2]}..., done={done}")
        pass  # Replace with actual learning logic


if __name__ == "__main__":

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        # You might also want to pass env_config for timestep duration etc.
        # env_config={'TIMESTEP_DURATION': {'seconds': 60}}
    )

    env.seed(0)

    # Instantiate your RL agent
    agent = MyRLAgent(env.observation_space, env.action_space)

    num_episodes = 3  # Let's run a few episodes

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1} ---")
        state = env.reset()  # Start a new episode
        done = False
        episode_reward = 0
        step_count = 0

        # Loop until the episode is done (market close or other termination)
        # The tqdm will track steps within one episode or over total steps if you change the range
        # Here, it's tracking steps within an episode.
        while not done:
            # 1. Agent chooses an action based on the current state
            action = agent.choose_action(state)

            # 2. Environment takes a step with the chosen action
            # Note: For Gym v0.26.0+, env.step returns (observation, reward, terminated, truncated, info)
            new_state, reward, done, info = env.step(action)
            print(f"New state: {state}, Reward: {reward}, Action: {action}, Done: {done}, Info: {info}")
            # 3. Agent learns from the experience
            agent.update_policy(state, action, reward, new_state, done)

            # Update current state and episode reward
            state = new_state
            episode_reward += reward
            step_count += 1

            # Optional: Add a break condition if the episode runs too long for testing
            if step_count > 500:  # Example: max 500 steps per episode for quick testing
                print("Episode truncated due to max steps for testing.")
                break

        print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {episode_reward:.2f}")

    env.close()
    print("Simulation finished.")