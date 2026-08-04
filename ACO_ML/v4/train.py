"""
Training Script with Curriculum Learning v4 (Fixed Resume Counter).
"""
import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ACO_ML.v4.config import (
    CHECKPOINT_DIR, TRAIN_INSTANCES_CURRICULUM, CONTROL_FREQ,
    BEST_KNOWN_SOLUTIONS
)
from ACO_ML.v4.data.tsp_loader import load_dataset_instance, TSPInstance
from ACO_ML.v4.data.solutions import get_gap
from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
from ACO_ML.v4.aco.aco_core import VectorizedACO
from ACO_ML.v4.agent.state_builder import build_state_vector
from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
from ACO_ML.v4.agent.ql_agent import QLearningAgent

def calculate_reward(prev_best: float, curr_best: float, base_scale: float, stagnation_cnt: int) -> float:
    imp = (prev_best - curr_best) / (base_scale + 1e-10)
    reward = imp * 100.0
    
    if imp <= 0:
        reward -= 0.05 * min(10, stagnation_cnt)
    else:
        reward += 1.0
        
    return float(reward)

def train_agents(
    max_episodes: int = 400,
    n_iter_per_ep: int = 150,
    control_freq: int = CONTROL_FREQ,
    debug: bool = False,
    resume: bool = False
):
    if debug:
        max_episodes = 10
        n_iter_per_ep = 30
        print("\n=== DEBUG MODE ENABLED (10 episodes, 30 iterations) [v4 Architecture] ===\n")
        
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ddpg_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt")
    ql_path = os.path.join(CHECKPOINT_DIR, "ql_agent.pkl")
    meta_path = os.path.join(CHECKPOINT_DIR, "train_meta.json")
    
    ddpg_agent = DDPGAgent()
    ql_agent = QLearningAgent()
    
    start_episode = 1
    
    if resume:
        if os.path.exists(ddpg_path):
            print(f"[RESUME] Loading DDPG Agent weights from {ddpg_path}...")
            ddpg_agent.load(ddpg_path)
        if os.path.exists(ql_path):
            print(f"[RESUME] Loading Q-Learning Agent weights from {ql_path}...")
            ql_agent.load(ql_path)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    start_episode = meta.get("last_episode", 0) + 1
                    print(f"[RESUME SUCCESS] Resuming from Episode {start_episode}/{max_episodes} (Previous knowledge preserved!)")
            except Exception:
                start_episode = 1
        else:
            # If meta file doesn't exist but weights exist (e.g. checkpoint 200 exists)
            start_episode = 201
            print(f"[RESUME SUCCESS] Resuming from Episode {start_episode}/{max_episodes} (Previous weights preserved!)")
        
    all_instance_names = ["kroA200", "pr439", "rat575", "u1060"]
    if debug:
        all_instance_names = ["kroA200"]
        
    instances = {}
    for name in all_instance_names:
        print(f"Loading dataset instance: {name}...")
        instances[name] = load_dataset_instance(name)
        
    start_time = time.time()
    ep_times = []
    
    print(f"\nStarting Curriculum Training v4 (Episodes {start_episode} to {max_episodes})...")
    print("=" * 70)
    
    for episode in range(start_episode, max_episodes + 1):
        ep_start = time.time()
        
        if debug:
            active_names = ["kroA200"]
        else:
            stage_idx = min(3, int(episode / (max_episodes / 4.0)))
            active_names = TRAIN_INSTANCES_CURRICULUM[stage_idx]
            
        inst_name = np.random.choice(active_names)
        instance = instances[inst_name]
        
        dynamic_env = DynamicTSPEnvironment(
            base_dist_matrix=instance.dist_matrix,
            perturb_ratio=0.10,
            perturb_amplitude=0.4,
            perturb_period=20,
            enabled=True,
            seed=episode
        )
        
        aco = VectorizedACO(
            dist_matrix=instance.dist_matrix,
            alpha=1.0, beta=5.0, rho=0.1,
            seed=episode
        )
        
        N = instance.n_cities
        base_scale = BEST_KNOWN_SOLUTIONS[inst_name]
        
        graph_stats = dynamic_env.get_change_stats()
        aco_stats = aco.get_search_stats()
        state_vec = build_state_vector(aco_stats, graph_stats, instance.dist_matrix, current_iter=0, max_iter=n_iter_per_ep)
        ql_state = ql_agent.discretize_state(aco_stats, current_iter=0, max_iter=n_iter_per_ep)
        
        raw_action, aco_params = ddpg_agent.select_action(state_vec, add_noise=True)
        ql_act_idx, ql_params = ql_agent.select_action(ql_state, evaluate=False)
        
        aco.set_parameters(aco_params["alpha"], aco_params["beta"], aco_params["rho"])
        
        c_loss_list, a_loss_list = [], []
        
        for it in range(1, n_iter_per_ep + 1):
            prev_best = aco.best_length
            
            active_dist = dynamic_env.step_environment()
            aco.set_dist_matrix(active_dist)
            
            iter_best = aco.step()
            curr_best = aco.best_length
            
            if it % control_freq == 0 or it == n_iter_per_ep:
                reward = calculate_reward(prev_best, curr_best, base_scale, aco.stagnation_counter)
                done = (it == n_iter_per_ep)
                
                graph_stats = dynamic_env.get_change_stats()
                aco_stats = aco.get_search_stats()
                next_state_vec = build_state_vector(aco_stats, graph_stats, active_dist, current_iter=it, max_iter=n_iter_per_ep)
                next_ql_state = ql_agent.discretize_state(aco_stats, current_iter=it, max_iter=n_iter_per_ep)
                
                ddpg_agent.memory.push(state_vec, raw_action, reward, next_state_vec, done)
                c_loss, a_loss = ddpg_agent.update()
                if c_loss > 0:
                    c_loss_list.append(c_loss)
                    a_loss_list.append(a_loss)
                    
                ql_agent.update(ql_state, ql_act_idx, reward, next_ql_state, done)
                
                state_vec = next_state_vec
                ql_state = next_ql_state
                
                raw_action, aco_params = ddpg_agent.select_action(state_vec, add_noise=True)
                ql_act_idx, ql_params = ql_agent.select_action(ql_state, evaluate=False)
                
                aco.set_parameters(aco_params["alpha"], aco_params["beta"], aco_params["rho"])
                
        ep_duration = time.time() - ep_start
        ep_times.append(ep_duration)
        
        final_gap = get_gap(aco.best_length, inst_name)
        
        c_loss_avg = np.mean(c_loss_list) if c_loss_list else 0.0
        a_loss_avg = np.mean(a_loss_list) if a_loss_list else 0.0
        
        print(f"Ep {episode:03d}/{max_episodes:03d} | Inst: {inst_name:7s} | Best: {aco.best_length:8.1f} | Gap: {final_gap:6.2f}% | C-Loss: {c_loss_avg:.4f} | A-Loss: {a_loss_avg:.4f} | Time: {ep_duration:.2f}s")
        
        if episode % 50 == 0 or episode == max_episodes:
            ddpg_agent.save(ddpg_path)
            ql_agent.save(ql_path)
            with open(meta_path, 'w') as f:
                json.dump({"last_episode": episode}, f)
            print(f"--> Saved checkpoint at episode {episode} to {CHECKPOINT_DIR}/")
            
        import gc
        gc.collect()
            
    total_min = (time.time() - start_time) / 60.0
    print("=" * 70)
    print(f"Training finished in {total_min:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes")
    parser.add_argument("--n_iter", type=int, default=150, help="ACO iterations per episode")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with minimal episodes")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()
    
    train_agents(
        max_episodes=args.episodes,
        n_iter_per_ep=args.n_iter,
        debug=args.debug,
        resume=args.resume
    )
