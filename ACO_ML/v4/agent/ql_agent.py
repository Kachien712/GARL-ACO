"""
Tuned Q-Learning Baseline Agent v4 for discrete ACO parameter control.
Does not use graph-aware features to provide a fair non-graph RL baseline.
"""
import os
import pickle
import numpy as np
from ACO_ML.v4.config import QL_CONFIG, ACTION_BOUNDS

class QLearningAgent:
    def __init__(self):
        alphas = [0.8, 1.5, 2.5]
        betas  = [2.0, 4.0, 5.5]
        rhos   = [0.1, 0.3, 0.6]
        
        self.action_space = []
        for a in alphas:
            for b in betas:
                for r in rhos:
                    self.action_space.append({"alpha": a, "beta": b, "rho": r})
                    
        self.n_actions = len(self.action_space)
        self.q_table = {}
        
        self.lr = QL_CONFIG["alpha"]
        self.gamma = QL_CONFIG["gamma"]
        self.epsilon = QL_CONFIG["epsilon_start"]
        self.epsilon_min = QL_CONFIG["epsilon_min"]
        self.epsilon_decay = QL_CONFIG["epsilon_decay"]

    def discretize_state(self, aco_stats: dict, current_iter: int, max_iter: int = 200) -> tuple:
        imp_5 = aco_stats.get("improvement_rate_5", 0.0)
        if imp_5 > 0.05:
            imp_bucket = 2
        elif imp_5 > 0.005:
            imp_bucket = 1
        else:
            imp_bucket = 0
            
        stag = aco_stats.get("stagnation_counter", 0)
        if stag < 5:
            stag_bucket = 0
        elif stag < 20:
            stag_bucket = 1
        else:
            stag_bucket = 2
            
        progress = current_iter / float(max_iter)
        if progress < 0.33:
            prog_bucket = 0
        elif progress < 0.66:
            prog_bucket = 1
        else:
            prog_bucket = 2
            
        return (imp_bucket, stag_bucket, prog_bucket)

    def select_action(self, state_tuple: tuple, evaluate: bool = False) -> tuple[int, dict]:
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.n_actions, dtype=np.float32)
            
        if not evaluate and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = int(np.argmax(self.q_table[state_tuple]))
            
        return action_idx, self.action_space[action_idx]

    def update(self, state_tuple: tuple, action_idx: int, reward: float, next_state_tuple: tuple, done: bool):
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.n_actions, dtype=np.float32)
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = np.zeros(self.n_actions, dtype=np.float32)
            
        current_q = self.q_table[state_tuple][action_idx]
        max_next_q = 0.0 if done else np.max(self.q_table[next_state_tuple])
        
        target_q = reward + self.gamma * max_next_q
        self.q_table[state_tuple][action_idx] += self.lr * (target_q - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"QLearning checkpoint {filepath} not found.")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
