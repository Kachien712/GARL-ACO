"""
Discrete-DQN Agent — Ablation B for VNICT 2026 Review 1.

Isolates the contribution of DDPG's CONTINUOUS action space from the
contribution of the 34-dim graph-aware state: this agent receives the exact
same 34-dim graph-aware state as GARL-ACO (ddpg_agent.py), but must choose
from the SAME 27 discrete (alpha, beta, rho) combinations as QLearningAgent
(imported directly from ql_agent.py so the discretization is byte-identical
across both baselines — a fair comparison).

Reading this alongside NoGraph-DDPG (state_builder.build_search_only_state_vector):
  GARL-ACO (full)  vs  NoGraph-DDPG   -> isolates graph-aware STATE contribution
  GARL-ACO (full)  vs  Discrete-DQN   -> isolates CONTINUOUS ACTION contribution
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from agent.state_builder import STATE_DIM
    from agent.ql_agent import QLearningAgent
    from agent.ddpg_agent import ReplayBuffer
except ImportError:
    from ACO_ML.v4.agent.state_builder import STATE_DIM
    from ACO_ML.v4.agent.ql_agent import QLearningAgent
    from ACO_ML.v4.agent.ddpg_agent import ReplayBuffer

device = torch.device("cpu")

class QNetwork(nn.Module):
    """Same width/depth as the DDPG Actor/Critic (34 -> 128 -> 128 -> N_actions with LayerNorm)
    so the ablations are architecturally comparable — only the output head differs
    (continuous action vs. per-discrete-action Q-values)."""
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = 27):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.ln1(self.fc1(state)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class DQNAgent:
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        gamma: float = 0.99,       # matches DDPG_CONFIG["gamma"] for a fair comparison
        tau: float = 0.005,        # matches DDPG_CONFIG["tau"] (soft target update)
        lr: float = 1e-3,          # matches DDPG_CONFIG["lr_critic"] (this IS a critic-only agent)
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        epsilon_start: float = 0.3,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
    ):
        # Re-use QLearningAgent's action table verbatim -> identical 27-way
        # discretization of (alpha, beta, rho) as the Q-Learning baseline.
        self._ql_ref = QLearningAgent()
        self.action_space = self._ql_ref.action_space
        self.n_actions = self._ql_ref.n_actions

        self.state_dim = state_dim
        self.q_net = QNetwork(state_dim, self.n_actions).to(device)
        self.q_target = QNetwork(state_dim, self.n_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def set_epsilon(self, epsilon: float):
        self.epsilon = float(epsilon)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> tuple[int, dict]:
        if not evaluate and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            self.q_net.eval()
            with torch.no_grad():
                q_values = self.q_net(state_tensor).squeeze(0).cpu().numpy()
            self.q_net.train()
            action_idx = int(np.argmax(q_values))
        return action_idx, self.action_space[action_idx]

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions_idx, rewards, next_states, dones = self.memory.sample(self.batch_size)
        actions_idx = actions_idx.long().view(-1, 1)  # action stored as scalar index, not raw_action vector

        current_q = self.q_net(states).gather(1, actions_idx)
        with torch.no_grad():
            next_q = self.q_target(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'q_target': self.q_target.state_dict(),
            'target_q_net': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"DQN checkpoint {filepath} not found.")
        checkpoint = torch.load(filepath, map_location=device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        target_dict = checkpoint.get('q_target') or checkpoint.get('target_q_net')
        if target_dict is not None:
            self.q_target.load_state_dict(target_dict)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
