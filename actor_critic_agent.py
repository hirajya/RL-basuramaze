import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ActorCritic, self).__init__()
        
        # Shared features
        self.shared_fc1 = nn.Linear(input_size, 128)
        self.shared_fc2 = nn.Linear(128, 64)
        
        # Actor head (policy)
        self.actor_fc1 = nn.Linear(64, 32)
        self.actor_fc2 = nn.Linear(32, n_actions)
        
        # Critic head (value function)
        self.critic_fc1 = nn.Linear(64, 32)
        self.critic_fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        shared_features = F.relu(self.shared_fc2(x))
        
        # Actor head
        actor_hidden = F.relu(self.actor_fc1(shared_features))
        action_probs = self.actor_fc2(actor_hidden)
        
        # Critic head
        critic_hidden = F.relu(self.critic_fc1(shared_features))
        state_value = self.critic_fc2(critic_hidden)
        
        return action_probs, state_value

class ActorCriticAgent:
    def __init__(self, settings, state_size):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_size, 4).to(self.device)
        
        # Separate optimizers for actor and critic
        # Shared layers + actor layers for actor optimizer
        actor_params = [
            self.network.shared_fc1.parameters(),
            self.network.shared_fc2.parameters(),
            self.network.actor_fc1.parameters(),
            self.network.actor_fc2.parameters()
        ]
        self.actor_optimizer = torch.optim.Adam(
            [p for params in actor_params for p in params],
            lr=self.settings.actor_lr
        )
        
        # Shared layers + critic layers for critic optimizer
        critic_params = [
            self.network.shared_fc1.parameters(),
            self.network.shared_fc2.parameters(),
            self.network.critic_fc1.parameters(),
            self.network.critic_fc2.parameters()
        ]
        self.critic_optimizer = torch.optim.Adam(
            [p for params in critic_params for p in params],
            lr=self.settings.critic_lr
        )
        
        # State value map for visualization
        self.state_values = {}
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_logits, state_value = self.network(state)
        
        # Store state value for visualization
        state_key = tuple(state.cpu().numpy().flatten())
        self.state_values[state_key] = state_value.item()
        
        # Apply temperature scaling
        scaled_logits = action_logits / self.settings.beta
        
        # Epsilon-greedy with softmax policy
        if np.random.random() < self.settings.epsilon:
            action = np.random.randint(4)
        else:
            probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            
        return action
        
    def update(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # Get current action logits and state value
        action_logits, state_value = self.network(state)
        _, next_value = self.network(next_state)
        
        # Compute target and advantage
        target = reward + (1 - done) * self.settings.gamma * next_value.detach()
        advantage = (target - state_value).detach()
        
        # Critic loss (value function)
        value_loss = self.settings.value_weight * F.mse_loss(state_value, target)
        
        # Actor loss (policy)
        probs = F.softmax(action_logits / self.settings.beta, dim=-1)
        log_prob = F.log_softmax(action_logits / self.settings.beta, dim=-1)[action]
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        actor_loss = -self.settings.advantage_weight * log_prob * advantage
        actor_loss -= self.settings.entropy_weight * entropy  # Add entropy regularization
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)  # Add retain_graph=True here
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def get_state_values(self):
        return self.state_values