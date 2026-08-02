"""
Deep Deterministic Policy Gradient (DDPG) Agent for continuous ACO parameter control.
Supports deterministic action selection for unbiased evaluation.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ACO_ML.v3.config import ACTION_BOUNDS, DDPG_CONFIG
from ACO_ML.v3.agent.state_builder import STATE_DIM

device = torch.device("cpu")

class OUNoise:
    """Ornstein-Uhlenbeck noise process for continuous action exploration."""
    def __init__(self, action_dimension: int, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dimension
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(self.action_dim)
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dim)

    def sample(self) -> np.ndarray:

        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=device),
            torch.tensor(np.array(action), dtype=torch.float32, device=device),
            torch.tensor(np.array(reward), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.tensor(np.array(done), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        # Raw action in [-1, 1]
        raw_action = self.tanh(self.fc3(x))
        return raw_action

class Critic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class DDPGAgent:
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DDPG_CONFIG["lr_actor"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DDPG_CONFIG["lr_critic"])
        
        self.memory = ReplayBuffer(DDPG_CONFIG["buffer_capacity"])
        self.noise = OUNoise(action_dim, theta=DDPG_CONFIG["ou_theta"], sigma=DDPG_CONFIG["ou_sigma"])
        
        self.gamma = DDPG_CONFIG["gamma"]
        self.tau = DDPG_CONFIG["tau"]
        self.batch_size = DDPG_CONFIG["batch_size"]

    @staticmethod
    def raw_to_aco_params(raw_action: np.ndarray) -> dict:
        """
        Maps raw neural output in [-1, 1] to exact ACO continuous parameter ranges:
            alpha in [0.5, 3.0]
            beta  in [1.0, 6.0]
            rho   in [0.05, 0.90]
        """
        a_min, a_max = ACTION_BOUNDS["alpha"]
        b_min, b_max = ACTION_BOUNDS["beta"]
        r_min, r_max = ACTION_BOUNDS["rho"]
        
        # Rescale from [-1, 1] to [0, 1]
        norm_a = (raw_action[0] + 1.0) / 2.0
        norm_b = (raw_action[1] + 1.0) / 2.0
        norm_r = (raw_action[2] + 1.0) / 2.0
        
        alpha = float(a_min + norm_a * (a_max - a_min))
        beta  = float(b_min + norm_b * (b_max - b_min))
        rho   = float(r_min + norm_r * (r_max - r_min))
        
        return {
            "alpha": np.clip(alpha, a_min, a_max),
            "beta":  np.clip(beta, b_min, b_max),
            "rho":   np.clip(rho, r_min, r_max),
        }

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> tuple[np.ndarray, dict]:
        """
        Select continuous action.
        
        Args:
            state: State vector (shape (27,))
            add_noise: If True, adds OU noise (training mode). If False, runs strictly deterministic (evaluation mode).
            
        Returns:
            raw_action: Raw numpy array in [-1, 1]^3
            aco_params: Dict of rescaled (alpha, beta, rho)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            raw_action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        self.actor.train()
        
        if add_noise:
            noise_val = self.noise.sample()
            raw_action = np.clip(raw_action + noise_val, -1.0, 1.0)
            
        aco_params = self.raw_to_aco_params(raw_action)
        return raw_action, aco_params

    def update(self) -> tuple[float, float]:
        """Update Actor and Critic networks using sampled replay batch."""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute Target Q
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            expected_q = rewards + (1.0 - dones) * self.gamma * target_q
            
        # Update Critic
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, expected_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update targets
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file {filepath} not found.")
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
