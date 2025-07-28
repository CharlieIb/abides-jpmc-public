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
    def __init__ (self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        # Shared layers for processing the state
        self.shared_layers = nn.Sequential (
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
        )

        # Actor head: outputs action probabilities (policy)
        self.actor_head = nn.Linear(64, output_dim)

        # Critic head: outputs a single state value
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        x = x.float()
        x = self.shared_layers(x)

        # Get action probabilities from the actor head
        action_probs = F.softmax(self.actor)

        # Get state value from the critic head
        state_value = self.critic_head(x)

        return action_probs, state_value


class PPOAgent():
    def __init__(self, observation_space, action_space,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                policy_clip=0.2, n_epoch=10, batch_size=64):
        self.gamma = gammaself.gae_lambda =gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        input_dim = int(np.prob(observation_space))
        action_dim = action_space.n

        self.policy = ActorCritic(input_dim, action_dim)
        self.optimiser = optim.Adam(self.policy.paramters(), lr=learning_rate)

        # Temporary memroy to store a batch of experiences
        self.memory = []

        print(f"Agent initialised for PPO. Obs space: {observation_space.shape}, Action space: {action_space.n}")

    def store_experience(self, state, action, log_prob, value, reward, done):
        """Stores a single step of experience."""
        self.memory.append((state, action, log_prob, value, reward, done))

    def choose_action(self, state):
        """Chooses an action based on the current policy."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            state_tensor = state_tensor.squeeze(-1)

            action_probs, state_value = self.policy(state_tensor)

            # Create a distribution to sample actions from
            dist = Categorical(action_probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_probl.item(), state_value.item()

    def update_policy(self):
        """Performs the PPO update step using the collected memory."""
        # Calculate Advantages using Generalised Advantage Estimation
        rewards = []
        discounted_reward = 0
        for _, _, _, _, reward, done in reversed(self.memory):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

            # Convert memory to tensors
            states, actions, old_log_probs, old_value, _, _, = zip(*self.memory)
            states = torch.tensor(np.array(states)).float()
            actions = torch.tensor(actions).long()
            old_log_probs = torch.tensor(old_log_probs).float()
            old_values = torch.tensor(old_values).float()
            rewards = torch.tensor(reward).float()

            # Calculate advantages A(t) = R(t) - V(s_t)
            advantages = rewards - old_values
            #Normalise advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. Optimise policy for K epochs
            for _ in range(self.n_epochs):
                # Create minibatches for the update
                for i in range(0, len(states), self.batch_size, len(states)):
                    batch_indices = np.arange(i, min(i + self.batch_size, len(states)))

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_rewards = rewards[batch_indices]

                    # Get new action probabilities and values from the current policy
                    new_probs, new_values = self.policy(batch_states)
                    new_dist = Categorical(new_probs)
                    new_log_probs = new_dist.log_prob(batch_actions)
                    entropy = new_dist.entropy()

                    # --- PPO Objective Function ---
                    # Ratio of new to old policy
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)

                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value function loss (MSE)
                    critic_loss = F.mse_loss(new_values.squeeze(), batch_rewards)

                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                    # Update the network
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()

                # Clear memory for the next batch
                self.memory = []



