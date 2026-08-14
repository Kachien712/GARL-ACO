"""
Generate publication-quality combined convergence curves comparing Fixed ACO, Q-Learning, and GARL-ACO on the same plot,
and print clean summary statistics to the terminal for easy analysis.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ACO_ML.v4.config import CHECKPOINT_DIR, BASE_DIR, BEST_KNOWN_SOLUTIONS
from ACO_ML.v4.data.tsp_loader import load_dataset_instance
from ACO_ML.v4.data.solutions import get_gap
from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
from ACO_ML.v4.agent.ql_agent import QLearningAgent
from ACO_ML.v4.baselines import run_fixed_aco, run_ql_aco, run_ddpg_aco
from ACO_ML.v4.evaluate import compute_recovery_iterations

PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def plot_combined_convergence(instance_name: str = "u1060", n_iter: int = 150, seed: int = 1000):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print(f"\n=====================================================================================", flush=True)
    print(f"  RUNNING CONVERGENCE COMPARISON: {instance_name} (Seed={seed}, Iterations={n_iter})", flush=True)
    print(f"=====================================================================================", flush=True)
    
    instance = load_dataset_instance(instance_name)
    best_known = BEST_KNOWN_SOLUTIONS.get(instance_name, None)
    
    ddpg_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt")
    ql_path = os.path.join(CHECKPOINT_DIR, "ql_agent.pkl")
    
    ddpg_agent = DDPGAgent()
    if os.path.exists(ddpg_path):
        ddpg_agent.load(ddpg_path)
    
    ql_agent = QLearningAgent()
    if os.path.exists(ql_path):
        ql_agent.load(ql_path)
        
    dynamic_env = DynamicTSPEnvironment(
        base_dist_matrix=instance.dist_matrix,
        perturb_ratio=0.12,
        perturb_amplitude=0.5,
        perturb_period=20,
        enabled=True,
        seed=seed
    )
    
    # Run all 3 models with identical dynamic environment seed
    print(f"  [1/3] Running Fixed ACO...", flush=True)
    fixed_res = run_fixed_aco(instance, n_iter=n_iter, dynamic_env=dynamic_env, seed=seed)
    
    print(f"  [2/3] Running ACO + Q-Learning...", flush=True)
    dynamic_env.reset()
    ql_res = run_ql_aco(instance, ql_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed)
    
    print(f"  [3/3] Running GARL-ACO...", flush=True)
    dynamic_env.reset()
    ddpg_res = run_ddpg_aco(instance, ddpg_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed)
    
    # Calculate summary stats for terminal display
    methods_data = [
        ("Fixed ACO (Baseline)", fixed_res),
        ("ACO + Q-Learning", ql_res),
        ("GARL-ACO (Proposed)", ddpg_res),
    ]
    
    print(f"\n-------------------------------------------------------------------------------------", flush=True)
    print(f"{'Method':22s} | {'Final Tour Length':18s} | {'Gap %':10s} | {'Recovery Metric':20s}", flush=True)
    print(f"-------------------------------------------------------------------------------------", flush=True)
    
    for name, res in methods_data:
        length = res["best_length"]
        gap = get_gap(length, instance_name)
        rec = compute_recovery_iterations(res["iter_best_curve"], perturb_step=20, recovery_threshold_ratio=1.05)
        rec_str = f"{rec:.1f} iters" if rec is not None else "Unrecovered"
        print(f"{name:22s} | {length:18.1f} | {gap:9.2f}% | {rec_str:20s}", flush=True)
        
    print(f"-------------------------------------------------------------------------------------\n", flush=True)
    
    # Plot combined graph
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    iters = range(1, n_iter + 1)
    ax.plot(iters, fixed_res["convergence_curve"], label="Fixed ACO (Baseline)", color="#e74c3c", lw=2, alpha=0.9)
    ax.plot(iters, ql_res["convergence_curve"], label="ACO + Q-Learning", color="#f39c12", lw=2, alpha=0.9)
    ax.plot(iters, ddpg_res["convergence_curve"], label="GARL-ACO", color="#2ecc71", lw=2.5)
    
    if best_known:
        ax.axhline(best_known, color="black", linestyle="--", lw=1.5, label=f"Best Known Opt ({best_known:,})")
        
    ax.set_title(f"Real-Time Convergence Comparison on {instance_name} under Dynamic Traffic", fontsize=13, fontweight='bold')
    ax.set_xlabel("ACO Iterations", fontsize=11)
    ax.set_ylabel("Best Tour Length", fontsize=11)
    
    # Format Y axis numbers with commas (e.g. 1,200,000 instead of 1.2 x 1e6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_path = os.path.join(PLOTS_DIR, f"convergence_comparison_{instance_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"--> Saved plot image to: {save_path}\n", flush=True)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default="all", help="Instance to evaluate (kroA200, pr439, rat575, u1060, pcb1173, or all)")
    parser.add_argument("--n_iter", type=int, default=150, help="ACO iterations per instance")
    args = parser.parse_args()
    
    if args.instance == "all":
        for inst in ["kroA200", "pr439", "rat575", "u1060", "pcb1173"]:
            plot_combined_convergence(instance_name=inst, n_iter=args.n_iter, seed=1000)
    else:
        plot_combined_convergence(instance_name=args.instance, n_iter=args.n_iter, seed=1000)
