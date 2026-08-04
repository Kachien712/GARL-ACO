"""
Plotting module v4 generating publication-ready comparison charts and parameter trajectories.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
    methods = ["Fixed ACO", "ACO + Q-Learning", "ACO + DDPG"]
    colors = {"Fixed ACO": "#e74c3c", "ACO + Q-Learning": "#f39c12", "ACO + DDPG": "#2ecc71"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(instances))
    width = 0.25
    
    for i, method in enumerate(methods):
        means = [data[inst][method]["mean_gap"] for inst in instances]
        stds  = [data[inst][method]["std_gap"] for inst in instances]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method, color=colors[method], capsize=4, alpha=0.9)
        
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("Benchmark Performance Comparison across TSPLIB Instances (v4 Architecture)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plot_path_bar = os.path.join(PLOTS_DIR, "gap_comparison_bar.png")
    plt.tight_layout()
    plt.savefig(plot_path_bar, dpi=300)
    plt.close()
    print(f"Saved gap comparison chart to {plot_path_bar}")

if __name__ == "__main__":
    generate_plots()
