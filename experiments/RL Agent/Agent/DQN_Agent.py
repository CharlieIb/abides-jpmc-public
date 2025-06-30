import gym
from tqdm import tqdm
import numpy as np
import random
from collections import deque

# Import deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import to register environments
import abides_gym

# --- Neural Network for DQN ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Using the architecture from the paper which was [50, 20]
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, output_dim)

    # Forward pass for the NN
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- RL Agent Class (DQN) ---
class DQNAgent():
    def __init__(self, observation_space, action_space,
                 learning_rate = 1e-3, discount_factor = 1.0,
                 exploration_start=1.0, exploration_end=0.02, exploration_decay_steps= 10000,
                 replay_buffer_size=10000, batch_size=32, target_update_freq=100
                 ):
        self.observation_space_shape = observation_space.shape
        self.action_space_n = action_space.n # The number of discrete actions (BUY, HOLD, SELL)

        # Policy Network: The network taht learns and is actively trained
        self.policy_net = DQN (int(np.prod(self.observation_space_shape)), self.action_space_n)
        # Target Network: A copy of the policy network, use for stable Q-value targets
        self.target_net = DQN(int(np.prod(self.observation_space_shape)), self.action_space_n)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Target net starts as copy of policy net
        self.target_net.eval() # Target network is usually in evaluation mode (no gradient updates)

        # Optimiser: How the network's weights will be updated during training
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Loss Function: Measures the difference between predicted and target Q-values
        self.criterion = nn.MSELoss() # Mean Squared Error loss for Q-value prediction

        # Hyperparameters for the RL algorithm
        self.discount_factor = discount_factor # Gamma (y): how much future rewards are values
        self.exploration_rate = exploration_start # Epsilon (e): current probability of random action
        self.exploration_end = exploration_end # Minimum epsilon value
        self.exploration_decay_steps = exploration_decay_steps
        # Rate at which epsilon linearly decreases per step
        self.epsilon_decay_linear = (exploration_start - exploration_end) / exploration_decay_steps

        # Replay Buffer: Stores past experiences (s, a, r, s', done)
        self.replay_buffer = deque(maxlen=replay_buffer_size) # deque is efficient for FIFO buffer
        self.batch_size = batch_size # Number of experiences to sample for each training update
        self.target_update_freq = target_update_freq # How often to update the target network
        self.learn_step_counter = 0 # Counter for learning steps, used for target network updates and exploration decay

        print(f"Agent initialized for DQN. Obs space: {observation_space.shape}, Action space: {action_space.n}")

    def choose_action(self, state):
        # Epsilon-greedy strategy
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            return random.randrange(self.action_space_n)
        else:
            # Exploit: choose action with highest predicted Q-value from the policy netowrk
            with torch.no_grad(): # Disable gradient calculations for inference
                # 1. Convert numpy state to torch tensor
                # state is typically a 1D numpy array (e.g., shape (7,))
                # .float() ensures it's a float tensor
                # .unsqueeze(0) adds a batch dimension (e.g., from (7,) to (1,7))
                # Neural networks in PyTorch typically expect batched inputs.
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                # 2. Get Q-values from the policy network
                # The network outputs Q-values for all possible actions for the given state
                q_values = self.policy_net(state_tensor)
                #3. Select the action with the maximum Q-value
                # .argmax(dim=1) returns the index of the maximum value along dimension 1 (the action dimension)
                # .item() extracts the single integer action value from the tensor
                return q_values.argmax(dim=1).item()

    def _store_experience(self, old_state, action, reward, new_state, done):
        """
        Stores a single experience tuple in the replay buffer.
        An experience tuple consists of (state, action reward, next_state, done_flag).
        """
        # Store(s, a, r, s', done) tuple in replay buffer
        self.replay_buffer.append((old_state, action, reward, new_state, done))


    def update_policy(self, old_state, action, reward, new_state, done):
        self.learn_step_counter += 1 # Increment total steps where learning could occur

        # 1. Store the current experience in the replay buffer
        self._store_experience(old_state, action, reward, new_state, done)

        # 2. Only start learning if the replay buffer has enough experiences for a batch
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough data yet, wait for more experiences

        # 3. Sample a random batch of experiences from the replay buffer
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        # Unpack the batch into separate lists of states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # 4. Convert lists of NumPy arrays/scalars to PyTorch Tensors
        # These will be the inputs for the neural network.
        states = torch.from_numpy(np.array(states)).float()
        # actions need to be Long (integer) type and unsqueezed for gather operation
        actions = torch.tensor(np.array(actions)).long().unsqueeze(1)
        # rewards and dones need to be float and unsqueezed for element-wise operations
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.tensor(dones).float().unsqueeze(1) # 'done' is boolean, convert to 0.0 or 1.0

        # 5. Compute Q-values for current states (Q(s,a)) using the Policy Network
        # 'self.policy_net(states)' outputs Q-values for all actions for each state in the batch.
        # '.gather(1, actions)' selects the Q-value specifically for the 'action' that was actually taken.
        # Example: if actions is [[0], [2], [1]], it picks the Q-value for action 0 from row 0, action 2 from row 1, etc.
        q_values = self.policy_net(states).gather(1, actions)

        # 6. Compute Max Q-value for next states (max(Q(s', a'))) using the Target Network
        # 'self.target_net(next_states)' outputs Q-values for all actions for each next_state in the batch.
        # '.max(1)' find the maximum Q-value along dimension 1 (actions) and returns (max_values, indices).
        # '[0]' selects only the max_values.
        # '.unsqueeze(1)' adds back the dimension for consistent shape for element-wise operations
        # '.detach()' is crucial: it stops gradients from flowing back into the target network
        # The target network is only updated by copying weights, not by gradient descent here.
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()

        # 7. Compute the Target Q-value (TD Target)
        # TD Target = Reward + gamma * max(Q(s', a')) * (1 - done_flag)
        # If 'done_flag' is True (1.0), then (1 - 1.0) = 0, so the future reward term vanishes
        # This correctly handles terminal states where there is no future.
        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        # 8. Compute the Loss
        # Measures the difference between the policy network's predicted Q-values
        # and the calculated target Q-values
        loss = self.criterion(q_values, target_q_values)

        # 9. Optimise the Policy Network
        # This is where the neural network's weights are actually updated.
        self.optimizer.zero_grad() # Clear gradients from previous optimisation step
        loss.backward() # Perform backpropagation: compute gradients of the loss w.r.t. policy_net's weights
        # Gradient Clipping (Optional but Recommended)
        # Prevents gradients from becoming too large, which can lead to unstable training.
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 10. Update Exploration Rate (Epsilon Decay)
        # Linearly decreases epsilon over time, balancing exploration and exploitation
        self.exploration_rate = max(self.exploration_end, self.exploration_rate - self.epsilon_decay_linear)

        # 11. Periodically Update the Target Network
        # Copy weights from the policy network to the target network at specified intervals
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated for {self.learn_step_counter} steps")


if __name__ == "__main__":

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        # Env config - alter it
        env_config={
            'ORDER_FIXED_SIZE': 100, # default size of market orders for BUY/SELL actions
            'TIMESTEP_DURATION': {'seconds': 60} # Agent wakes up every minute
        }
    )

    env.seed(0)


    # Instantiate the RL-DQN agent
    # Get the shape of the observation space (e.g. (7,)) and number of actions (e.g. ,3)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-3,
        discount_factor=1,
        exploration_start=1.0,
        exploration_end=0.02,
        exploration_decay_steps=10000, # Tune as needed
        replay_buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
    )


    # Training Loop Parameters
    num_episodes = 50  # Tune as needed

    # Track total steps for exploration decay (if using a global decay schedule)
    total_steps = 0


    # Episode Interation

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1} ---")

        # Reset Environment for a New Episode
        # Expecting 1 return value for Gym 0.21.0 - 0.25.x
        state = env.reset()  # Start a new episode
        done = False
        episode_reward = 0
        step_count = 0

        # Loop until the episode is done (market close or other termination)
        # The tqdm will track steps within one episode or over total steps if you change the range
        # Here, it's tracking steps within an episode.
        with tqdm(total=None, desc=f"Episode {episode+1}", unit="steps") as pbar:
            while not done:
                # 1. Agent chooses an action based on the current state
                action = agent.choose_action(state)

                # 2. Environment takes a step with the chosen action
                # For Gym v0.21.0 - 0.25.x, env.step returns (obs, reward, done, info)
                # Note: For Gym v0.26.0+, env.step returns (observation, reward, terminated, truncated, info)
                new_state, reward, done, info = env.step(action)

                # 3. Agent learns from the experience
                agent.update_policy(state, action, reward, new_state, done)

                # Update current state and episode reward
                state = new_state
                episode_reward += reward
                step_count += 1
                total_steps += 1 # Global step counter for exploration/logging

                pbar.update(1) # Update tqdm progress bar

                # Optional: Add a break condition if the episode runs too long for testing
                if step_count > 1000:  # Example: max 500 steps per episode for quick testing
                    print("Episode truncated due to max steps for testing.")
                    done = True # Force done if this happens to exit loop
                    break

        print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {episode_reward:.2f}")

    env.close()
    print("Simulation finished.")