"""
Unified Training Framework v4.4 for GARL-ACO, NoGraph-DDPG, Discrete-DQN, and Tabular Q-Learning.
(VNICT 2026 Rigorous Experimental Revision).

CRITICAL METHODOLOGICAL GUARANTEE:
  Every RL agent (DDPG, NoGraph-DDPG, Discrete-DQN, Q-Learning) is trained on its OWN
  INDEPENDENT ACO rollout with identical initial conditions and perturbation schedules.
  NO action/reward leakage across agents — each agent is 100% responsible for the ACO
  trajectory and rewards used for its updates.
"""
import os
import sys
import time
import json
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from config import (
        CHECKPOINT_DIR, TRAIN_INSTANCES_CURRICULUM, CONTROL_FREQ,
        BEST_KNOWN_SOLUTIONS, DDPG_CONFIG, STATE_DIM, SEARCH_ONLY_STATE_DIM
    )
    from data.tsp_loader import load_dataset_instance, TSPInstance
    from data.solutions import get_gap
    from env.dynamic_tsp_env import DynamicTSPEnvironment
    from aco.aco_core import VectorizedACO
    from agent.state_builder import build_state_vector, build_search_only_state_vector
    from agent.ddpg_agent import DDPGAgent
    from agent.ql_agent import QLearningAgent
    from agent.dqn_agent import DQNAgent
except ImportError:
    from ACO_ML.v4.config import (
        CHECKPOINT_DIR, TRAIN_INSTANCES_CURRICULUM, CONTROL_FREQ,
        BEST_KNOWN_SOLUTIONS, DDPG_CONFIG, STATE_DIM, SEARCH_ONLY_STATE_DIM
    )
    from ACO_ML.v4.data.tsp_loader import load_dataset_instance, TSPInstance
    from ACO_ML.v4.data.solutions import get_gap
    from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
    from ACO_ML.v4.aco.aco_core import VectorizedACO
    from ACO_ML.v4.agent.state_builder import build_state_vector, build_search_only_state_vector
    from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
    from ACO_ML.v4.agent.ql_agent import QLearningAgent
    from ACO_ML.v4.agent.dqn_agent import DQNAgent

def calculate_reward(
    prev_best: float,
    curr_best: float,
    base_scale: float,
    stagnation_cnt: int
) -> float:
    imp = (prev_best - curr_best) / (base_scale + 1e-10)
    reward = imp * 100.0
    if imp <= 0:
        reward -= 0.05 * min(10, stagnation_cnt)
    else:
        reward += 1.0
    return float(reward)

def _run_single_agent_episode(
    agent,
    agent_type: str,  # "ddpg_full" | "ddpg_nograph" | "dqn" | "ql"
    instance: TSPInstance,
    n_iter_per_ep: int,
    control_freq: int,
    episode_seed: int,
    base_scale: float
) -> tuple[float, float, float]:
    """
    Executes a completely independent ACO rollout for a specific agent.
    Guarantees no state or reward leakage between different methods.
    """
    # 1. Independent dynamic environment & ACO instance
    dynamic_env = DynamicTSPEnvironment(
        base_dist_matrix=instance.dist_matrix,
        perturb_ratio=0.12,
        perturb_amplitude=0.4,
        perturb_period=20,
        enabled=True,
        seed=episode_seed
    )
    aco = VectorizedACO(
        dist_matrix=instance.dist_matrix,
        alpha=1.0, beta=5.0, rho=0.1,
        seed=episode_seed
    )

    def _get_state(cur_it, dist_mat):
        graph_stats = dynamic_env.get_change_stats()
        aco_stats = aco.get_search_stats()
        if agent_type == "ddpg_full" or agent_type == "dqn":
            return build_state_vector(aco_stats, graph_stats, dist_mat, current_iter=cur_it, max_iter=n_iter_per_ep)
        elif agent_type == "ddpg_nograph":
            return build_search_only_state_vector(aco_stats, dist_mat, current_iter=cur_it, max_iter=n_iter_per_ep)
        elif agent_type == "ql":
            return agent.discretize_state(aco_stats, current_iter=cur_it, max_iter=n_iter_per_ep)

    # 2. Initial action selection
    state = _get_state(0, instance.dist_matrix)
    if agent_type in ("ddpg_full", "ddpg_nograph"):
        action, aco_params = agent.select_action(state, add_noise=True)
    elif agent_type == "dqn":
        action, aco_params = agent.select_action(state, evaluate=False)
    elif agent_type == "ql":
        action, aco_params = agent.select_action(state, evaluate=False)

    aco.set_parameters(aco_params["alpha"], aco_params["beta"], aco_params["rho"])
    loss_list = []

    # 3. Step through episode iterations
    for it in range(1, n_iter_per_ep + 1):
        prev_best = aco.best_length

        active_dist = dynamic_env.step_environment()
        aco.set_dist_matrix(active_dist)

        iter_best = aco.step()
        curr_best = aco.best_length

        if it % control_freq == 0 or it == n_iter_per_ep:
            reward = calculate_reward(prev_best, curr_best, base_scale, aco.stagnation_counter)
            done = (it == n_iter_per_ep)

            next_state = _get_state(it, active_dist)

            # Agent updates its own policy based on its own reward
            if agent_type in ("ddpg_full", "ddpg_nograph"):
                agent.memory.push(state, action, reward, next_state, done)
                c_loss, a_loss = agent.update()
                if c_loss > 0:
                    loss_list.append(c_loss)
                next_action, next_params = agent.select_action(next_state, add_noise=True)
            elif agent_type == "dqn":
                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.update()
                if loss > 0:
                    loss_list.append(loss)
                next_action, next_params = agent.select_action(next_state, evaluate=False)
            elif agent_type == "ql":
                agent.update(state, action, reward, next_state, done)
                next_action, next_params = agent.select_action(next_state, evaluate=False)

            state = next_state
            action = next_action
            aco.set_parameters(next_params["alpha"], next_params["beta"], next_params["rho"])

    final_gap = get_gap(aco.best_length, instance.name)
    avg_loss = float(np.mean(loss_list)) if loss_list else 0.0
    return final_gap, float(aco.best_length), avg_loss

def train_ablation_agents(
    target_agents: list = None,
    max_episodes: int = 250,
    n_iter_per_ep: int = 100,
    control_freq: int = CONTROL_FREQ,
    resume: bool = False
):
    if target_agents is None:
        target_agents = ["ddpg_full", "ddpg_nograph", "dqn", "ql"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    all_instance_names = ["kroA200", "pr439", "rat575", "u1060"]
    instances = {name: load_dataset_instance(name) for name in all_instance_names}

    print("=================== INDEPENDENT MULTI-AGENT TRAINING v4.4 ===================")
    print(f"Target Agents to Train: {target_agents}")
    print(f"Episodes: {max_episodes} | Iterations per Episode: {n_iter_per_ep} | Control Freq: {control_freq}")
    print(f"Curriculum: {TRAIN_INSTANCES_CURRICULUM}")
    print("===============================================================================\n")

    # Initialize Agents
    agents_dict = {}
    checkpoint_paths = {
        "ddpg_full": os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt"),
        "ddpg_nograph": os.path.join(CHECKPOINT_DIR, "ddpg_agent_nograph.pt"),
        "dqn": os.path.join(CHECKPOINT_DIR, "dqn_agent.pt"),
        "ql": os.path.join(CHECKPOINT_DIR, "ql_agent.pkl"),
    }

    if "ddpg_full" in target_agents:
        agents_dict["ddpg_full"] = DDPGAgent(state_dim=STATE_DIM)
        if resume and os.path.exists(checkpoint_paths["ddpg_full"]):
            agents_dict["ddpg_full"].load(checkpoint_paths["ddpg_full"])
            print(f"[RESUME] Loaded ddpg_full from {checkpoint_paths['ddpg_full']}")

    if "ddpg_nograph" in target_agents:
        agents_dict["ddpg_nograph"] = DDPGAgent(state_dim=SEARCH_ONLY_STATE_DIM)
        if resume and os.path.exists(checkpoint_paths["ddpg_nograph"]):
            agents_dict["ddpg_nograph"].load(checkpoint_paths["ddpg_nograph"])
            print(f"[RESUME] Loaded ddpg_nograph from {checkpoint_paths['ddpg_nograph']}")

    if "dqn" in target_agents:
        agents_dict["dqn"] = DQNAgent(state_dim=STATE_DIM)
        if resume and os.path.exists(checkpoint_paths["dqn"]):
            agents_dict["dqn"].load(checkpoint_paths["dqn"])
            print(f"[RESUME] Loaded dqn from {checkpoint_paths['dqn']}")

    if "ql" in target_agents:
        agents_dict["ql"] = QLearningAgent()
        if resume and os.path.exists(checkpoint_paths["ql"]):
            agents_dict["ql"].load(checkpoint_paths["ql"])
            print(f"[RESUME] Loaded ql from {checkpoint_paths['ql']}")

    OU_SIGMA_START = DDPG_CONFIG["ou_sigma_start"]
    OU_SIGMA_END = DDPG_CONFIG["ou_sigma_end"]

    start_time = time.time()

    for episode in range(1, max_episodes + 1):
        ep_t0 = time.time()

        # Noise & Exploration schedule
        decay_progress = (episode - 1) / float(max(1, max_episodes - 1))
        current_sigma = OU_SIGMA_START + (OU_SIGMA_END - OU_SIGMA_START) * decay_progress
        current_epsilon = max(0.01, 0.35 * (1.0 - decay_progress * 0.9))

        for k in ["ddpg_full", "ddpg_nograph"]:
            if k in agents_dict:
                agents_dict[k].set_noise_sigma(current_sigma)
                agents_dict[k].noise.reset()

        if "dqn" in agents_dict:
            agents_dict["dqn"].set_epsilon(current_epsilon)

        # Curriculum stage
        stage_idx = min(3, int((episode - 1) / (max_episodes / 4.0)))
        active_names = TRAIN_INSTANCES_CURRICULUM[stage_idx]
        inst_name = np.random.choice(active_names)
        instance = instances[inst_name]
        base_scale = BEST_KNOWN_SOLUTIONS[inst_name]

        # Execute independent rollout for each agent
        ep_stats = {}
        for agent_key in target_agents:
            gap, best_len, loss = _run_single_agent_episode(
                agent=agents_dict[agent_key],
                agent_type=agent_key,
                instance=instance,
                n_iter_per_ep=n_iter_per_ep,
                control_freq=control_freq,
                episode_seed=episode * 1000 + 42,
                base_scale=base_scale
            )
            ep_stats[agent_key] = {"gap": gap, "loss": loss}

        ep_duration = time.time() - ep_t0

        # Console logging
        stat_strs = []
        for k in target_agents:
            stat_strs.append(f"{k}: {ep_stats[k]['gap']:5.1f}%")
        print(f"Ep {episode:03d}/{max_episodes:03d} (Stage {stage_idx+1}) | {inst_name:7s} | {' | '.join(stat_strs)} | Sigma={current_sigma:.3f}, Eps={current_epsilon:.3f} | {ep_duration:.2f}s")

        # Checkpointing
        if episode % 50 == 0 or episode == max_episodes:
            for k in target_agents:
                agents_dict[k].save(checkpoint_paths[k])
            print(f"--> Saved verified checkpoints at Episode {episode} to {CHECKPOINT_DIR}/")

    total_min = (time.time() - start_time) / 60.0
    print(f"\nAll targeted agents trained independently in {total_min:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=str, default="all",
                        choices=["all", "dqn", "ql", "ddpg", "nograph"],
                        help="Which agent(s) to train with independent rollouts")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--n_iter", type=int, default=100, help="ACO iterations per episode")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    agent_map = {
        "all": ["ddpg_full", "ddpg_nograph", "dqn", "ql"],
        "dqn": ["dqn"],
        "ql": ["ql"],
        "ddpg": ["ddpg_full"],
        "nograph": ["ddpg_nograph"]
    }

    train_ablation_agents(
        target_agents=agent_map[args.agents],
        max_episodes=args.episodes,
        n_iter_per_ep=args.n_iter,
        resume=args.resume
    )
