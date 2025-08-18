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

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, output_dim)

    # Forward pass for the NN
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent():
    def __init__(self, observation_space, action_space,
                 learning_rate = 1e-3, discount_factor = 1.0,
                 exploration_start=1.0, exploration_end=0.02, exploration_decay_steps= 10000,
                 replay_buffer_size=10000, batch_size=32, target_update_freq=100,
                 use_confidence_sizing: bool = False
                 ):
        self.observation_space_shape = observation_space.shape
        self.action_space_n = action_space.n

        # Policy Network: The network that learns and is actively trained
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
        self.exploration_start = exploration_start
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

        self.use_confidence_sizing = use_confidence_sizing

        self.avg_long_entry_price = 0.0
        self.avg_short_entry_price = 0.0

        self.profit_exits = 0
        self.loss_exits = 0
        self.total_trades_closed = 0

        print(f"Agent initialized for DQN. Obs space: {observation_space.shape}, Action space: {action_space.n}")

    def reset_agent_state(self):
        """
        Resets the agent's state for a new episode.
        This is primarily used to reset the exploration rate.
        """
        self.avg_long_entry_price = 0.0
        self.avg_short_entry_price = 0.0
        self.profit_exits = 0
        self.loss_exits = 0
        self.total_trades_closed = 0



    def choose_action(self, state, info=None):

        if info and 'action_mask' in info:
            action_mask = info['action_mask']
        else:
            action_mask = np.ones(self.action_space_n, dtype=np.int8)

        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return (0, 0.0) if self.use_confidence_sizing else 0

        # Epsilon-greedy strategy
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            action = np.random.choice(valid_actions)

            if self.use_confidence_sizing:
                return action, 0.5 # Return tuple with neutral confidence
            else:
                return action
        else:
            # Exploit: choose action with highest predicted Q-value from the policy network
            with torch.no_grad(): # Disable gradient calculations for inference
                # 1. Convert numpy state to torch tensor
                # state is typically a 1D numpy array (e.g., shape (7,))
                # .float() ensures it's a float tensor
                # .unsqueeze(0) adds a batch dimension (e.g., from (7,) to (1,7))
                # Neural networks in PyTorch typically expect batched inputs.
                state_tensor = torch.from_numpy(state.flatten()).float().unsqueeze(0)
                # 2. Get Q-values from the policy network
                # The network outputs Q-values for all possible actions for the given state
                q_values = self.policy_net(state_tensor).squeeze(0)
                #3. Select the action with the maximum Q-value
                # .argmax(dim=1) returns the index of the maximum value along dimension 1 (the action dimension)
                # .item() extracts the single integer action value from the tensor

                # Apply the mask: set Q-values of invalid actions to -inf
                # This ensures they are never chosen by argmax
                masked_q_values = q_values.clone()
                masked_q_values[action_mask == 0] = -float('inf')

                # Select the action with the maximum Q-value from the masked net
                action = masked_q_values.argmax().item()
                if self.use_confidence_sizing:
                    action_probs = F.softmax(masked_q_values, dim=0)
                    confidence = action_probs.max().item()
                    return action, confidence
                else:
                    return action

    def _store_experience(self, old_state, action, reward, new_state, done):
        """
        Stores a single experience tuple in the replay buffer.
        An experience tuple consists of (state, action reward, next_state, done_flag).
        """
        # Store(s, a, r, s', done) tuple in replay buffer
        self.replay_buffer.append((old_state, action, reward, new_state, done))


    def update_policy(self, old_state, action_or_tuple, reward, new_state, done):

        # Get holdings from the state vectors (assuming holdings are at index 5)
        old_holdings = old_state[5][0]
        new_holdings = new_state[5][0]

        # Using VWAP as price estimate
        current_price = old_state[0][0]
        exit_price = new_state[0][0]

        # LONG position changes
        if old_holdings >= 0 and new_holdings > 0:
            trade_size = new_holdings - old_holdings
            # Opening a new long position
            if old_holdings == 0:
                self.avg_long_entry_price = current_price
            # Scaling into an existing long position
            elif new_holdings > old_holdings:
                new_avg_price = ((old_holdings * self.avg_long_entry_price) + (
                            trade_size * current_price)) / new_holdings
                self.avg_long_entry_price = new_avg_price

        # SHORT Position changes
        elif old_holdings <= 0 and new_holdings < 0:
            trade_size = abs(new_holdings - old_holdings)
            # Opening a new short position
            if old_holdings == 0:
                self.avg_short_entry_price = current_price
            # Scaling into an existing short position
            elif new_holdings < old_holdings:
                new_avg_price = ((abs(old_holdings) * self.avg_short_entry_price) + (trade_size * current_price)) / abs(
                    new_holdings)
                self.avg_short_entry_price = new_avg_price

        # EXITS
        # Selling part of a LONG position
        if old_holdings > 0 and new_holdings < old_holdings:
            sold_size = old_holdings - new_holdings
            pnl = (exit_price - self.avg_long_entry_price) * sold_size
            if pnl > 0:
                self.profit_exits += 1
            else:
                self.loss_exits += 1
            self.total_trades_closed += 1
            # Position fully closed, reset entry price
            if new_holdings == 0:
                self.avg_long_entry_price = 0.0
                print(f"DEBUG: Position closed. P&L: {pnl:.2f}. Wins: {self.profit_exits}, Losses: {self.loss_exits}")

        # Buying back part or all of a short position
        if old_holdings < 0 and new_holdings > old_holdings:
            covered_size = abs(old_holdings - new_holdings)
            pnl = (self.avg_short_entry_price - exit_price) * covered_size
            if pnl > 0:
                self.profit_exits += 1
            else:
                self.loss_exits += 1
            self.total_trades_closed += 1
            if new_holdings == 0:
                # position fully closed reset entry price
                self.avg_short_entry_price = 0.0
                print(f"DEBUG: Position closed. P&L: {pnl:.2f}. Wins: {self.profit_exits}, Losses: {self.loss_exits}")

        if self.use_confidence_sizing:
            action, _ = action_or_tuple
        else:
            action = action_or_tuple

        self.learn_step_counter += 1 # Increment total steps where learning could occur

        # Store the current experience
        self._store_experience(old_state, action, reward, new_state, done)

        # Check it is time to learn
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a random batch
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to tensors
        states = torch.from_numpy(np.array(states)).float()
        states = states.squeeze(-1)
        actions = torch.tensor(np.array(actions)).long().unsqueeze(1)
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states).squeeze()).float()
        dones = torch.tensor(dones).float().unsqueeze(1)

        # Compute Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute Max Q-value for next states (max(Q(s', a'))) using the Target Network
        # 'self.target_net(next_states)' outputs Q-values for all actions for each next_state in the batch.
        # '.max(1)' find the maximum Q-value along dimension 1 (actions) and returns (max_values, indices).
        # '[0]' selects only the max_values.
        # '.unsqueeze(1)' adds back the dimension for consistent shape for element-wise operations
        # '.detach()' is crucial: it stops gradients from flowing back into the target network
        # The target network is only updated by copying weights, not by gradient descent here.
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()

        # Compute the Target Q-value (TD Target)
        # TD Target = Reward + gamma * max(Q(s', a')) * (1 - done_flag)
        # If 'done_flag' is True (1.0), then (1 - 1.0) = 0, so the future reward term vanishes
        # This correctly handles terminal states where there is no future.
        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        # Compute the Loss
        loss = self.criterion(q_values, target_q_values)

        # Optimise the Policy Network
        # This is where the neural network's weights are actually updated.
        self.optimizer.zero_grad() # Clear gradients from previous optimisation step
        loss.backward() # Perform backpropagation: compute gradients of the loss w.r.t. policy_net's weights
        # Gradient Clipping (Optional but Recommended)
        # Prevents gradients from becoming too large, which can lead to unstable training.
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update Exploration Rate (Epsilon Decay)
        # Linearly decreases epsilon over time, balancing exploration and exploitation
        self.exploration_rate = max(self.exploration_end, self.exploration_rate - self.epsilon_decay_linear)

        # Periodically Update the Target Network
        # Copy weights from the policy network to the target network at specified intervals
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print(f"Target network updated for {self.learn_step_counter} steps")

    def save_weights(self, file_path: str):
        """Saves the policy network's weights to a file."""
        print(f"\nSaving weights to {file_path}...")
        torch.save(self.policy_net.state_dict(), file_path)
        print("Weights saved.")

    def load_weights(self, file_path: str):
        """Loads weights into the policy and target networks from a file."""
        print(f"\nLoading weights from {file_path}...")
        self.policy_net.load_state_dict(torch.load(file_path))
        # Crucially, also update the target network to match
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()  # Set to evaluation mode
        self.target_net.eval()
        print("Weights loaded.")

    def get_episode_diagnostics(self):
        """Returns a dictionary of the episode's performance."""
        win_rate = self.profit_exits / self.total_trades_closed if self.total_trades_closed > 0 else 0
        return {
            "total_trades_closed": self.total_trades_closed,
            "win_rate": win_rate,
            "profit_exits": self.profit_exits,
            "loss_exits": self.loss_exits,
            "time_exits": 0  # Not applicable to this agent
        }


