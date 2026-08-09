import os

# Project Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataset"))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Best-known tour lengths for TSPLIB benchmark instances
BEST_KNOWN_SOLUTIONS = {
    "kroA200": 29368,
    "pr439": 107217,
    "rat575": 6773,
    "u1060": 224094,
    "pcb1173": 56892,  # Held-out test set
    "gr229": 134602
}

# Curriculum learning progression
TRAIN_INSTANCES_CURRICULUM = [
    ["kroA200"],
    ["kroA200", "pr439"],
    ["kroA200", "pr439", "rat575"],
    ["kroA200", "pr439", "rat575", "u1060"]
]
TEST_INSTANCE = "pcb1173"

# ACO Hyperparameter bounds for continuous RL action space
ACTION_BOUNDS = {
    "alpha": (0.5, 3.0),
    "beta": (1.0, 6.0),
    "rho": (0.05, 0.90),
}

# Default Fixed ACO parameters
FIXED_ACO_PARAMS = {
    "alpha": 1.0,
    "beta": 5.0,
    "rho": 0.1,
}

# DDPG Hyperparameters
DDPG_CONFIG = {
    "lr_actor": 1e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_capacity": 50000,
    "batch_size": 64,
    "ou_theta": 0.15,
    "ou_sigma": 0.2,
}

# Q-Learning Baseline Hyperparameters
QL_CONFIG = {
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon_start": 0.3,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.99,
}

# Environment & Control Settings
CONTROL_FREQ = 5          # Number of ACO iterations per RL decision step
DEFAULT_N_ITER = 200      # Standard evaluation iterations per instance
