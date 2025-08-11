import numpy as np

# Import deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# Import to register environments
import abides_gym
# Actor-Critic Neural Netowrk for PPO
class ActorCritic(nn.Module):
    def __init__ (self, input_dim, output_dim, hidden_size=128):
        super(ActorCritic, self).__init__()

        # Shared layers for processing the state
        self.shared_layers = nn.Sequential (
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        # Actor head: outputs action probabilities (policy)
        self.actor_head = nn.Linear(hidden_size // 2 , output_dim)

        # Critic head: outputs a single state value
        self.critic_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = x.float()
        x = self.shared_layers(x)

        # Get action probabilities from the actor head
        action_probs = F.softmax(self.actor_head(x), dim=-1)

        # Get state value from the critic head
        state_value = self.critic_head(x)

        return action_probs, state_value


class PPOAgent():
    def __init__(self, observation_space, action_space,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                policy_clip=0.2, n_epochs=10, batch_size=64,
                 learn_interval=2048, use_confidence_sizing=False
                 ):
        self.gamma = gamma
        self.gae_lambda =gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learn_interval = learn_interval
        self.use_confidence_sizing=use_confidence_sizing

        input_dim = int(np.prod(observation_space.shape))
        action_dim = action_space.n

        self.action_space_size = action_space.n # for the action mask

        self.policy = ActorCritic(input_dim, action_dim)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Temporary memory to store a batch of experiences
        self.memory = []

        print(f"Agent initialised for PPO. Obs space: {observation_space.shape}, Action space: {action_space.n}")

    def reset_agent_state(self):
        """Resets memory at the start of the new episode"""
        self.memory = []

    def store_experience(self, state, action, log_prob, value, reward, done):
        """Stores a single step of experience."""
        self.memory.append((state, action, log_prob, value, reward, done))

    def choose_action(self, state, info=None):
        """Chooses an action based on the current policy."""
        if info and 'action_mask' in info:
            action_mask = torch.tensor(info['action_mask'], dtype=torch.float32)
        else:
            action_mask = torch.ones(self.action_space_size)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.flatten()).float().unsqueeze(0)
            action_probs, state_value = self.policy(state_tensor)

            masked_probs = action_probs * action_mask
            if masked_probs.sum() == 0:
                # If all valid actions have zero probability, fall back to a uniform distribution over valid actions
                valid_actions = (action_mask == 1).nonzero(as_tuple=True)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions.numpy())
                    return (action, 0.5) if self.use_confidence_sizing else action
                else:
                    return (0, 0.0) if self.use_confidence_sizing else 0

            masked_probs /= masked_probs.sum()


            # Create a distribution to sample actions from
            dist = Categorical(masked_probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)

        self.store_experience(state.flatten(), action.item(), log_prob.item(), state_value.item(), 0, False)

        if self.use_confidence_sizing:
            confidence = masked_probs[0, action.item()].item()
            return action.item(), confidence
        else:
            return action.item()


    def update_policy(self, state, action_or_tuple, reward, new_state, done):
        """
        We update the last memory entry with the correct reward and done flag.
        The actual learning happens in a separate 'learn' method called periodically.
        """
        if not self.memory:
            return
        # Get the last stored experience and update it with the reward and the done status
        last_exp = list(self.memory[-1])
        last_exp[4] = reward
        last_exp[5] = done
        self.memory[-1] = tuple(last_exp)

    def learn(self):

        if not self.memory:
            return

        states, actions, old_log_probs, values, rewards, dones = zip(*self.memory)
        values = torch.tensor(values, dtype=torch.float32)

        advantages = []
        last_advantage = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i+1]

            mask = 1.0 - dones[i]
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
            advantages.insert(0, last_advantage)

            # Calculate advantages A(t) = R(t) - V(s_t)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        #Normalise advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert other memory to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)

        # 3. Optimise policy for K epochs
        indices = np.arange(len(self.memory))
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            # Create minibatches for the update
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get new action probabilities and values from the current policy
                new_probs, new_values = self.policy(states[batch_indices])
                new_dist = Categorical(new_probs)
                new_log_probs = new_dist.log_prob(actions[batch_indices])
                entropy = new_dist.entropy()

                # --- PPO Objective Function ---
                # Ratio of new to old policy
                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])

                # Clipped surrogate objective
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch_indices]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                critic_loss = F.mse_loss(new_values.squeeze(-1), returns[batch_indices])

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy.mean()

                # Update the network
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

        # Clear memory for the next batch
        self.memory = []

    def save_weights(self, file_path: str):
        """Saves the policy network's weights to a file."""
        print(f"\nSaving weights to {file_path}...")
        torch.save(self.policy.state_dict(), file_path)
        print("Weights saved.")

    def load_weights(self, file_path: str):
        """Loads weights into the policy and target networks from a file."""
        print(f"\nLoading weights from {file_path}...")
        self.policy.load_state_dict(torch.load(file_path))
        self.policy.eval()  # Set to evaluation mode
        print("Weights loaded.")



