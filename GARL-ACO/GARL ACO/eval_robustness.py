"""
Dynamic Robustness Evaluation Suite (VNICT 2026 Camera-Ready Revision).
Evaluates GARL-ACO vs Fixed ACO and Q-Learning under:
  1. Multi-Intensity Perturbations: 5%, 12% (default), 20%, 30% edge perturbations.
  2. Multi-Frequency Dynamics: Period T = 10, 20 (default), 40 ACO iterations.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from config import CHECKPOINT_DIR, BEST_KNOWN_SOLUTIONS, DEFAULT_N_ITER, STATE_DIM
    from data.tsp_loader import load_dataset_instance
    from data.solutions import get_gap, get_best_known
    from env.dynamic_tsp_env import DynamicTSPEnvironment
    from agent.ddpg_agent import DDPGAgent
    from agent.ql_agent import QLearningAgent
    from baselines import run_fixed_aco, run_ql_aco, run_ddpg_aco
except ImportError:
    from ACO_ML.v4.config import CHECKPOINT_DIR, BEST_KNOWN_SOLUTIONS, DEFAULT_N_ITER, STATE_DIM
    from ACO_ML.v4.data.tsp_loader import load_dataset_instance
    from ACO_ML.v4.data.solutions import get_gap, get_best_known
    from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
    from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
    from ACO_ML.v4.agent.ql_agent import QLearningAgent
    from ACO_ML.v4.baselines import run_fixed_aco, run_ql_aco, run_ddpg_aco

PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def run_robustness_experiments(
    instances: list = None,
    n_iter: int = 150,
    n_runs: int = 5
):
    if instances is None:
        instances = ["kroA200", "pr439", "pcb1173"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    ddpg_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt")
    ql_path = os.path.join(CHECKPOINT_DIR, "ql_agent.pkl")

    ddpg_agent = DDPGAgent(state_dim=STATE_DIM)
    ddpg_agent.load(ddpg_path)
    ql_agent = QLearningAgent()
    ql_agent.load(ql_path)

    intensities = [0.05, 0.12, 0.20, 0.30]
    frequencies = [10, 20, 40]

    robustness_data = {
        "intensity_experiments": {},
        "frequency_experiments": {}
    }

    print("=================== DYNAMIC ROBUSTNESS EXPERIMENT SUITE ===================")
    print(f"Instances: {instances}")
    print(f"Intensities: {[f'{int(p*100)}%' for p in intensities]}")
    print(f"Periods: {frequencies}")
    print(f"Iterations: {n_iter} | Runs per condition: {n_runs}")
    print("===========================================================================\n")

    # --- EXPERIMENT A: Multi-Intensity Perturbations ---
    print(">>> Running Experiment A: Perturbation Intensity Variation (Period=20, Amplitude=0.4)...")
    for inst_name in instances:
        instance = load_dataset_instance(inst_name)
        robustness_data["intensity_experiments"][inst_name] = {}

        for p_ratio in intensities:
            p_label = f"{int(p_ratio * 100)}%"
            res_p = {"Fixed ACO": [], "ACO + Q-Learning": [], "GARL-ACO": []}

            for run_idx in range(n_runs):
                seed = 2000 + run_idx
                dyn_env = DynamicTSPEnvironment(
                    base_dist_matrix=instance.dist_matrix,
                    perturb_ratio=p_ratio,
                    perturb_amplitude=0.4,
                    perturb_period=20,
                    enabled=True,
                    seed=seed
                )

                r_fixed = run_fixed_aco(instance, n_iter=n_iter, dynamic_env=dyn_env, seed=seed)
                r_ql = run_ql_aco(instance, ql_agent, n_iter=n_iter, dynamic_env=dyn_env, evaluate=True, seed=seed)
                r_garl = run_ddpg_aco(instance, ddpg_agent, n_iter=n_iter, dynamic_env=dyn_env, evaluate=True, seed=seed, state_mode="full")

                res_p["Fixed ACO"].append(get_gap(r_fixed["best_length"], inst_name))
                res_p["ACO + Q-Learning"].append(get_gap(r_ql["best_length"], inst_name))
                res_p["GARL-ACO"].append(get_gap(r_garl["best_length"], inst_name))

            summary_p = {}
            for m in ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO"]:
                summary_p[m] = {
                    "mean_gap": float(np.mean(res_p[m])),
                    "std_gap": float(np.std(res_p[m]))
                }
            robustness_data["intensity_experiments"][inst_name][p_label] = summary_p
            print(f"  [{inst_name}] Intensity {p_label:4s} -> Fixed: {summary_p['Fixed ACO']['mean_gap']:6.2f}% | QL: {summary_p['ACO + Q-Learning']['mean_gap']:6.2f}% | GARL: {summary_p['GARL-ACO']['mean_gap']:6.2f}%")

    # --- EXPERIMENT B: Multi-Frequency Dynamics ---
    print("\n>>> Running Experiment B: Perturbation Frequency Variation (Intensity=12%, Amplitude=0.4)...")
    for inst_name in instances:
        instance = load_dataset_instance(inst_name)
        robustness_data["frequency_experiments"][inst_name] = {}

        for period in frequencies:
            freq_label = f"Period_{period}"
            res_f = {"Fixed ACO": [], "ACO + Q-Learning": [], "GARL-ACO": []}

            for run_idx in range(n_runs):
                seed = 3000 + run_idx
                dyn_env = DynamicTSPEnvironment(
                    base_dist_matrix=instance.dist_matrix,
                    perturb_ratio=0.12,
                    perturb_amplitude=0.4,
                    perturb_period=period,
                    enabled=True,
                    seed=seed
                )

                r_fixed = run_fixed_aco(instance, n_iter=n_iter, dynamic_env=dyn_env, seed=seed)
                r_ql = run_ql_aco(instance, ql_agent, n_iter=n_iter, dynamic_env=dyn_env, evaluate=True, seed=seed)
                r_garl = run_ddpg_aco(instance, ddpg_agent, n_iter=n_iter, dynamic_env=dyn_env, evaluate=True, seed=seed, state_mode="full")

                res_f["Fixed ACO"].append(get_gap(r_fixed["best_length"], inst_name))
                res_f["ACO + Q-Learning"].append(get_gap(r_ql["best_length"], inst_name))
                res_f["GARL-ACO"].append(get_gap(r_garl["best_length"], inst_name))

            summary_f = {}
            for m in ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO"]:
                summary_f[m] = {
                    "mean_gap": float(np.mean(res_f[m])),
                    "std_gap": float(np.std(res_f[m]))
                }
            robustness_data["frequency_experiments"][inst_name][freq_label] = summary_f
            print(f"  [{inst_name}] Frequency {freq_label:10s} -> Fixed: {summary_f['Fixed ACO']['mean_gap']:6.2f}% | QL: {summary_f['ACO + Q-Learning']['mean_gap']:6.2f}% | GARL: {summary_f['GARL-ACO']['mean_gap']:6.2f}%")

    out_file = os.path.join(CHECKPOINT_DIR, "robustness_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(robustness_data, f, indent=2)
    print(f"\nSaved robustness results to {out_file}")

    # Plotting Curves
    _plot_robustness(robustness_data, instances, intensities, frequencies)
    return robustness_data

def _plot_robustness(data: dict, instances: list, intensities: list, frequencies: list):
    colors = {"Fixed ACO": "#e74c3c", "ACO + Q-Learning": "#f39c12", "GARL-ACO": "#2ecc71"}
    markers = {"Fixed ACO": "s", "ACO + Q-Learning": "^", "GARL-ACO": "o"}

    # Figure 1: Intensity
    fig, axes = plt.subplots(1, len(instances), figsize=(15, 4.5), sharey=False)
    if len(instances) == 1:
        axes = [axes]

    p_labels = [f"{int(p * 100)}%" for p in intensities]

    for idx, inst in enumerate(instances):
        ax = axes[idx]
        for m in ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO"]:
            means = [data["intensity_experiments"][inst][p][m]["mean_gap"] for p in p_labels]
            stds = [data["intensity_experiments"][inst][p][m]["std_gap"] for p in p_labels]
            ax.errorbar(p_labels, means, yerr=stds, label=m, color=colors[m], marker=markers[m], capsize=4, linewidth=1.8)

        ax.set_title(f"Instance: {inst}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Perturbation Ratio (% edges)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Optimality Gap (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle("Dynamic Robustness: Performance under Varying Perturbation Intensities", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_p = os.path.join(PLOTS_DIR, "robustness_intensity.png")
    plt.savefig(plot_p, dpi=300)
    plt.close()
    print(f"Saved intensity robustness plot to {plot_p}")

    # Figure 2: Frequency
    fig, axes = plt.subplots(1, len(instances), figsize=(15, 4.5), sharey=False)
    if len(instances) == 1:
        axes = [axes]

    f_labels = [f"Period_{f}" for f in frequencies]
    x_axis_labels = [f"T={f}" for f in frequencies]

    for idx, inst in enumerate(instances):
        ax = axes[idx]
        for m in ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO"]:
            means = [data["frequency_experiments"][inst][f][m]["mean_gap"] for f in f_labels]
            stds = [data["frequency_experiments"][inst][f][m]["std_gap"] for f in f_labels]
            ax.errorbar(x_axis_labels, means, yerr=stds, label=m, color=colors[m], marker=markers[m], capsize=4, linewidth=1.8)

        ax.set_title(f"Instance: {inst}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Perturbation Period (iterations)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Optimality Gap (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle("Dynamic Robustness: Performance under Varying Perturbation Frequencies", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_f = os.path.join(PLOTS_DIR, "robustness_frequency.png")
    plt.savefig(plot_f, dpi=300)
    plt.close()
    print(f"Saved frequency robustness plot to {plot_f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=150)
    parser.add_argument("--n_runs", type=int, default=5)
    args = parser.parse_args()
    run_robustness_experiments(n_iter=args.n_iter, n_runs=args.n_runs)
