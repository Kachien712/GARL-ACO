"""
Evaluation Script for Standardized Benchmark Comparison across TSPLIB Instances.
Computes gap%, std dev, and recovery metrics under dynamic traffic perturbations.
"""
import os
import sys
import json
import argparse
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ACO_ML.v3.config import (

    CHECKPOINT_DIR, BEST_KNOWN_SOLUTIONS, DEFAULT_N_ITER, TEST_INSTANCE
)
from ACO_ML.v3.data.tsp_loader import load_dataset_instance
from ACO_ML.v3.data.solutions import get_gap, get_best_known
from ACO_ML.v3.env.dynamic_tsp_env import DynamicTSPEnvironment
from ACO_ML.v3.agent.ddpg_agent import DDPGAgent
from ACO_ML.v3.agent.ql_agent import QLearningAgent
from ACO_ML.v3.baselines import run_fixed_aco, run_ql_aco, run_ddpg_aco

def compute_recovery_iterations(
    convergence_curve: list,
    perturb_step: int,
    recovery_threshold_ratio: float = 1.05
) -> float:
    """
    Measures iterations required to recover tour quality after graph perturbation at perturb_step.
    
    If no recovery occurs within the remaining iterations, returns float('inf') to cleanly distinguish 
    unrecovered runs from measured numerical recoveries.
    """
    if perturb_step >= len(convergence_curve):
        return 0.0
        
    pre_perturb_length = convergence_curve[perturb_step - 1]
    target_length = pre_perturb_length * recovery_threshold_ratio
    
    for idx in range(perturb_step, len(convergence_curve)):
        if convergence_curve[idx] <= target_length:
            return float(idx - perturb_step + 1)
            
    # Explicit unrecovered marker (infinity)
    return float('inf')

def evaluate_benchmark(
    n_iter: int = DEFAULT_N_ITER,
    n_runs: int = 3,
    instances_to_eval: list = None,
    output_json: str = "evaluation_results.json"
):
    if instances_to_eval is None:
        instances_to_eval = ["kroA200", "pr439", "rat575", "u1060", "pcb1173"]
        
    print(f"\n=================== EVALUATION SUITE ===================")
    print(f"Iterations per Instance: {n_iter} (Consistent for all instances)")
    print(f"Number of Evaluation Seeds: {n_runs}")
    print(f"Test Set (Zero-Shot): {TEST_INSTANCE}")
    print(f"========================================================\n")
    
    ddpg_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt")
    ql_path = os.path.join(CHECKPOINT_DIR, "ql_agent.pkl")
    
    ddpg_agent = DDPGAgent()
    if os.path.exists(ddpg_path):
        ddpg_agent.load(ddpg_path)
        print(f"Loaded DDPG Agent checkpoint from {ddpg_path}")
    else:
        print(f"[WARNING] DDPG checkpoint not found at {ddpg_path}! Running with initial weights.")
        
    ql_agent = QLearningAgent()
    if os.path.exists(ql_path):
        ql_agent.load(ql_path)
        print(f"Loaded Q-Learning Agent checkpoint from {ql_path}")
    else:
        print(f"[WARNING] Q-Learning checkpoint not found at {ql_path}! Running with initial weights.")
        
    results_summary = {}
    
    for inst_name in instances_to_eval:
        print(f"\n>>> Evaluating Instance: {inst_name} (Best Known: {get_best_known(inst_name)})")
        instance = load_dataset_instance(inst_name)
        
        inst_results = {
            "Fixed ACO": {"tour_lengths": [], "gaps": [], "curves": [], "recovery_iters": []},
            "ACO + Q-Learning": {"tour_lengths": [], "gaps": [], "curves": [], "recovery_iters": []},
            "ACO + DDPG": {"tour_lengths": [], "gaps": [], "curves": [], "param_trajectories": [], "recovery_iters": []},
        }
        
        for run_idx in range(n_runs):
            seed = 1000 + run_idx
            
            # Dynamic Traffic Environment for evaluation
            dynamic_env = DynamicTSPEnvironment(
                base_dist_matrix=instance.dist_matrix,
                perturb_ratio=0.10,
                perturb_amplitude=0.4,
                perturb_period=30,
                enabled=True,
                seed=seed
            )
            
            # 1. Fixed ACO
            fixed_res = run_fixed_aco(instance, n_iter=n_iter, dynamic_env=dynamic_env, seed=seed)
            gap_fixed = get_gap(fixed_res["best_length"], inst_name)
            rec_fixed = compute_recovery_iterations(fixed_res["convergence_curve"], perturb_step=30)
            inst_results["Fixed ACO"]["tour_lengths"].append(fixed_res["best_length"])
            inst_results["Fixed ACO"]["gaps"].append(gap_fixed)
            inst_results["Fixed ACO"]["curves"].append(fixed_res["convergence_curve"])
            inst_results["Fixed ACO"]["recovery_iters"].append(rec_fixed)
            
            # 2. ACO + Q-Learning
            ql_res = run_ql_aco(instance, ql_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed)
            gap_ql = get_gap(ql_res["best_length"], inst_name)
            rec_ql = compute_recovery_iterations(ql_res["convergence_curve"], perturb_step=30)
            inst_results["ACO + Q-Learning"]["tour_lengths"].append(ql_res["best_length"])
            inst_results["ACO + Q-Learning"]["gaps"].append(gap_ql)
            inst_results["ACO + Q-Learning"]["curves"].append(ql_res["convergence_curve"])
            inst_results["ACO + Q-Learning"]["recovery_iters"].append(rec_ql)
            
            # 3. ACO + DDPG (Strictly Deterministic actor, add_noise=False inside evaluate=True)
            ddpg_res = run_ddpg_aco(instance, ddpg_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed)
            gap_ddpg = get_gap(ddpg_res["best_length"], inst_name)
            rec_ddpg = compute_recovery_iterations(ddpg_res["convergence_curve"], perturb_step=30)
            inst_results["ACO + DDPG"]["tour_lengths"].append(ddpg_res["best_length"])
            inst_results["ACO + DDPG"]["gaps"].append(gap_ddpg)
            inst_results["ACO + DDPG"]["curves"].append(ddpg_res["convergence_curve"])
            inst_results["ACO + DDPG"]["param_trajectories"].append(ddpg_res["param_trajectory"])
            inst_results["ACO + DDPG"]["recovery_iters"].append(rec_ddpg)
            
        # Print summary for this instance
        print("-" * 65)
        print(f"Method              | Mean Tour Length    | Mean Gap %        | Std Gap %")
        print("-" * 65)
        for method in ["Fixed ACO", "ACO + Q-Learning", "ACO + DDPG"]:
            mean_len = float(np.mean(inst_results[method]["tour_lengths"]))
            mean_gap = float(np.mean(inst_results[method]["gaps"]))
            std_gap  = float(np.std(inst_results[method]["gaps"]))
            
            # Handle recovery metric cleanly
            rec_list = inst_results[method]["recovery_iters"]
            valid_recs = [r for r in rec_list if r != float('inf')]
            unrec_cnt = len(rec_list) - len(valid_recs)
            rec_str = f"{np.mean(valid_recs):.1f} iters" if valid_recs else "Unrecovered"
            if unrec_cnt > 0:
                rec_str += f" ({unrec_cnt}/{len(rec_list)} unrecovered)"
                
            print(f"{method:19s} | {mean_len:17.1f} | {mean_gap:15.2f}% | {std_gap:8.2f}% | Rec: {rec_str}")
            
        results_summary[inst_name] = inst_results
        
    # Save benchmark results to JSON
    out_path = os.path.join(CHECKPOINT_DIR, output_json)
    
    # Convert numpy types before JSON serialization
    json_safe_dict = {}
    for inst, m_dict in results_summary.items():
        json_safe_dict[inst] = {}
        for m, data in m_dict.items():
            json_safe_dict[inst][m] = {
                "tour_lengths": [float(x) for x in data["tour_lengths"]],
                "gaps": [float(x) for x in data["gaps"]],
                "mean_gap": float(np.mean(data["gaps"])),
                "std_gap": float(np.std(data["gaps"])),
                "mean_tour": float(np.mean(data["tour_lengths"])),
                "recovery_iters": [str(x) if x == float('inf') else float(x) for x in data["recovery_iters"]]
            }
            
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe_dict, f, indent=2)
        
    print(f"\nSaved raw evaluation summary to {out_path}")
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=DEFAULT_N_ITER, help="Iterations per instance")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of random seeds per instance")
    args = parser.parse_args()
    
    evaluate_benchmark(n_iter=args.n_iter, n_runs=args.n_runs)
