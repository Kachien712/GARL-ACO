"""
Plotting Module v4.3 (VNICT 2026 Camera-Ready Revision).
Generates publication-quality charts:
  1. gap_comparison_bar.png: Linear scale Optimality Gap (%) with Std error bars.
  2. gap_comparison_log.png: Log-scale Optimality Gap (%) for clear visibility across all methods.
  3. control_overhead_latency.png: Trade-off between Decision Latency (ms) and Solution Quality.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from config import CHECKPOINT_DIR, BASE_DIR
except ImportError:
    from ACO_ML.v4.config import CHECKPOINT_DIR, BASE_DIR

PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def generate_plots(results_json_path: str = None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if results_json_path is None:
        results_json_path = os.path.join(CHECKPOINT_DIR, "evaluation_results.json")
        
    if not os.path.exists(results_json_path):
        print(f"Error: Evaluation results JSON file not found at {results_json_path}. Run evaluate.py first.")
        return
        
    with open(results_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    instances = list(data.keys())
    all_methods = ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO", "NoGraph-DDPG", "Discrete-DQN"]
    colors = {
        "Fixed ACO": "#e74c3c",
        "ACO + Q-Learning": "#f39c12",
        "GARL-ACO": "#2ecc71",
        "NoGraph-DDPG": "#3498db",   # Ablation A (Search features only)
        "Discrete-DQN": "#9b59b6",   # Ablation B (Discrete actions)
    }
    methods = [m for m in all_methods if all(m in data[inst] for inst in instances)]

    # 1. Figure 1: Linear Scale Gap Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(instances))
    width = 0.8 / max(1, len(methods))

    for i, method in enumerate(methods):
        means = [data[inst][method]["mean_gap"] for inst in instances]
        stds  = [data[inst][method]["std_gap"] for inst in instances]
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method, color=colors[method], capsize=4, alpha=0.9)
        
    ax.set_ylabel("Optimality Gap (%)", fontsize=12, fontweight='bold')
    ax.set_title("Benchmark Performance Comparison across TSPLIB Instances (Linear Scale)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plot_path_bar = os.path.join(PLOTS_DIR, "gap_comparison_bar.png")
    plt.tight_layout()
    plt.savefig(plot_path_bar, dpi=300)
    plt.close()
    print(f"Saved gap comparison chart (linear) to {plot_path_bar}")

    # 2. Figure 2: Log Scale Gap Comparison (Reviewer recommendation)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(methods):
        means = [data[inst][method]["mean_gap"] for inst in instances]
        stds  = [data[inst][method]["std_gap"] for inst in instances]
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method, color=colors[method], capsize=4, alpha=0.9, log=True)

    ax.set_ylabel("Optimality Gap (%) [Log Scale]", fontsize=12, fontweight='bold')
    ax.set_title("Benchmark Performance Comparison across TSPLIB Instances (Log-Y Scale)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path_log = os.path.join(PLOTS_DIR, "gap_comparison_log.png")
    plt.tight_layout()
    plt.savefig(plot_path_log, dpi=300)
    plt.close()
    print(f"Saved gap comparison chart (log-scale) to {plot_path_log}")

    # 3. Figure 3: Quality-vs-Cost Trade-off (Latency vs Gap)
    _generate_overhead_chart(data, instances, methods, colors)

def _generate_overhead_chart(data: dict, instances: list, methods: list, colors: dict):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(instances))
    width = 0.8 / max(1, len(methods))

    for i, method in enumerate(methods):
        latencies = [data[inst][method]["mean_control_latency_ms"] for inst in instances]
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax1.bar(x + offset, latencies, width, label=method, color=colors[method], alpha=0.85)

    ax1.set_ylabel("Mean Control-Decision Latency (ms)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("TSPLIB Benchmark Instance", fontsize=12, fontweight='bold')
    ax1.set_title("Computational Latency per Control Decision Step (Quality-vs-Cost Trade-off)", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(instances, fontsize=11)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path = os.path.join(PLOTS_DIR, "control_overhead_latency.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved control overhead/latency chart to {plot_path}")

if __name__ == "__main__":
    generate_plots()
