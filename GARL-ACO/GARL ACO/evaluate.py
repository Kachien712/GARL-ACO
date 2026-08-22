"""
Evaluation Script v4.3 for GARL-ACO — VNICT 2026 Camera-Ready Revision.

Key Features:
  1. Statistical Reliability: Computes Mean +/- Std, 95% Confidence Intervals (CI_95), and Wilcoxon signed-rank p-values.
  2. Fine-Grained Computational Timing: Measures T_ACO, T_state, T_infer, T_control, T_total, and decision latency.
  3. Randomized per-seed method execution order to eliminate thermal/systemic OS drift.
  4. Full compatibility with Ablations (NoGraph-DDPG / DDPG-Search, Discrete-DQN) and Zero-Shot generalization (pcb1173).
"""
import os
import sys
import json
import argparse
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from config import (
        CHECKPOINT_DIR, BEST_KNOWN_SOLUTIONS, DEFAULT_N_ITER, TEST_INSTANCE,
        STATE_DIM, SEARCH_ONLY_STATE_DIM, GRAPH_ONLY_STATE_DIM
    )
    from data.tsp_loader import load_dataset_instance
    from data.solutions import get_gap, get_best_known
    from env.dynamic_tsp_env import DynamicTSPEnvironment
    from agent.ddpg_agent import DDPGAgent
    from agent.ql_agent import QLearningAgent
    from agent.dqn_agent import DQNAgent
    from baselines import run_fixed_aco, run_ql_aco, run_ddpg_aco, run_dqn_aco
except ImportError:
    from ACO_ML.v4.config import (
        CHECKPOINT_DIR, BEST_KNOWN_SOLUTIONS, DEFAULT_N_ITER, TEST_INSTANCE,
        STATE_DIM, SEARCH_ONLY_STATE_DIM, GRAPH_ONLY_STATE_DIM
    )
    from ACO_ML.v4.data.tsp_loader import load_dataset_instance
    from ACO_ML.v4.data.solutions import get_gap, get_best_known
    from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
    from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
    from ACO_ML.v4.agent.ql_agent import QLearningAgent
    from ACO_ML.v4.agent.dqn_agent import DQNAgent
    from ACO_ML.v4.baselines import run_fixed_aco, run_ql_aco, run_ddpg_aco, run_dqn_aco

METHOD_LABELS = ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO", "NoGraph-DDPG", "Discrete-DQN"]
REFERENCE_METHOD = "GARL-ACO"

def compute_recovery_iterations(
    iter_best_curve: list,
    perturb_step: int = 30,
    recovery_threshold_ratio: float = 1.05
):
    if iter_best_curve is None or len(iter_best_curve) <= perturb_step:
        return None

    start_idx = max(0, perturb_step - 5)
    pre_perturb_baseline = float(np.mean(iter_best_curve[start_idx:perturb_step]))
    target_length = pre_perturb_baseline * recovery_threshold_ratio

    spike_detected = False
    for idx in range(perturb_step, len(iter_best_curve)):
        if iter_best_curve[idx] > target_length:
            spike_detected = True
            break

    if not spike_detected:
        return 0.0

    for idx in range(perturb_step, len(iter_best_curve)):
        if iter_best_curve[idx] <= target_length:
            return float(idx - perturb_step + 1)

    return None

def compute_significance(gaps_a: list, gaps_b: list) -> dict:
    a = np.asarray(gaps_a, dtype=np.float64)
    b = np.asarray(gaps_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("compute_significance requires paired same-length, same-seed-order samples")

    n = len(a)
    diffs = a - b

    if np.all(diffs == 0):
        return {
            "test_used": "identical (all paired differences are zero)",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "n_pairs": int(n),
            "min_possible_p_wilcoxon": float(2.0 / (2 ** n)) if n > 0 else None,
        }

    test_used = "Wilcoxon signed-rank"
    try:
        if n < 6:
            raise ValueError("sample too small for a meaningful Wilcoxon test")
        stat, p_value = stats.wilcoxon(a, b)
    except Exception:
        test_used = "Paired Student's t-test (fallback)"
        stat, p_value = stats.ttest_rel(a, b)

    return {
        "test_used": test_used,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "n_pairs": int(n),
        "min_possible_p_wilcoxon": float(2.0 / (2 ** n)) if n > 0 else None,
    }

def _load_or_fail(agent, path: str, label: str, allow_missing: bool):
    if os.path.exists(path):
        agent.load(path)
        print(f"[LOADED] {label}: {path}")
        return True
    msg = f"[MISSING CHECKPOINT] {label} not found at {path}"
    if allow_missing:
        print(f"[WARNING] {msg} — continuing with randomly-initialized weights.")
        return False
    raise FileNotFoundError(
        f"{msg}. Aborting to avoid benchmarking random weights. Pass --allow_missing_checkpoints if intentional."
    )

def evaluate_benchmark(
    n_iter: int = DEFAULT_N_ITER,
    n_runs: int = 8,
    instances_to_eval: list = None,
    output_json: str = "evaluation_results.json",
    allow_missing_checkpoints: bool = False,
    skip_ablations: bool = False,
):
    if instances_to_eval is None:
        instances_to_eval = ["kroA200", "pr439", "rat575", "u1060", "pcb1173"]

    active_methods = ["Fixed ACO", "ACO + Q-Learning", "GARL-ACO"]
    if not skip_ablations:
        active_methods += ["NoGraph-DDPG", "Discrete-DQN"]

    print(f"\n=================== GARL-ACO EVALUATION SUITE v4.3 ===================")
    print(f"Iterations per Instance: {n_iter}")
    print(f"Number of Evaluation Seeds (n_runs): {n_runs}")
    print(f"Test Set (Zero-Shot): {TEST_INSTANCE}")
    print(f"Methods: {active_methods}")
    print(f"Per-seed method execution order is RANDOMIZED.")
    print(f"========================================================================\n")

    ddpg_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt")
    ql_path = os.path.join(CHECKPOINT_DIR, "ql_agent.pkl")
    nograph_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent_nograph.pt")
    dqn_path = os.path.join(CHECKPOINT_DIR, "dqn_agent.pt")

    ddpg_agent = DDPGAgent(state_dim=STATE_DIM)
    _load_or_fail(ddpg_agent, ddpg_path, "GARL-ACO DDPG Agent", allow_missing_checkpoints)

    ql_agent = QLearningAgent()
    _load_or_fail(ql_agent, ql_path, "Q-Learning Agent", allow_missing_checkpoints)

    nograph_agent, dqn_agent = None, None
    if not skip_ablations:
        nograph_agent = DDPGAgent(state_dim=SEARCH_ONLY_STATE_DIM)
        _load_or_fail(nograph_agent, nograph_path, "NoGraph-DDPG Ablation Agent", allow_missing_checkpoints)

        dqn_agent = DQNAgent(state_dim=STATE_DIM)
        _load_or_fail(dqn_agent, dqn_path, "Discrete-DQN Ablation Agent", allow_missing_checkpoints)

    results_summary = {}

    for inst_name in instances_to_eval:
        print(f"\n>>> Evaluating Instance: {inst_name} (Best Known: {get_best_known(inst_name)})")
        instance = load_dataset_instance(inst_name)

        inst_results = {
            m: {
                "tour_lengths": [], "gaps": [], "curves": [], "recovery_iters": [],
                "total_runtimes": [], "aco_runtimes": [], "state_overheads": [], "infer_overheads": [],
                "control_overheads": [], "control_latencies_ms": [], "n_control_calls": [],
            } for m in active_methods
        }
        if "GARL-ACO" in inst_results:
            inst_results["GARL-ACO"]["param_trajectories"] = []

        for run_idx in range(n_runs):
            seed = 1000 + run_idx

            order_rng = np.random.RandomState(999_000 + seed)
            run_order = list(active_methods)
            order_rng.shuffle(run_order)

            dynamic_env = DynamicTSPEnvironment(
                base_dist_matrix=instance.dist_matrix,
                perturb_ratio=0.12,
                perturb_amplitude=0.4,
                perturb_period=20,
                enabled=True,
                seed=seed
            )

            run_fns = {
                "Fixed ACO": lambda: run_fixed_aco(instance, n_iter=n_iter, dynamic_env=dynamic_env, seed=seed),
                "ACO + Q-Learning": lambda: run_ql_aco(instance, ql_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed),
                "GARL-ACO": lambda: run_ddpg_aco(instance, ddpg_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed, state_mode="full"),
                "NoGraph-DDPG": lambda: run_ddpg_aco(instance, nograph_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed, state_mode="search_only"),
                "Discrete-DQN": lambda: run_dqn_aco(instance, dqn_agent, n_iter=n_iter, dynamic_env=dynamic_env, evaluate=True, seed=seed),
            }

            for method in run_order:
                res = run_fns[method]()
                gap = get_gap(res["best_length"], inst_name)
                rec = compute_recovery_iterations(res["iter_best_curve"], perturb_step=20, recovery_threshold_ratio=1.05)

                bucket = inst_results[method]
                bucket["tour_lengths"].append(res["best_length"])
                bucket["gaps"].append(gap)
                bucket["curves"].append(res["convergence_curve"])
                bucket["recovery_iters"].append(rec)
                bucket["total_runtimes"].append(res["total_runtime_sec"])
                bucket["aco_runtimes"].append(res.get("aco_runtime_sec", res["total_runtime_sec"]))
                bucket["state_overheads"].append(res.get("state_overhead_sec", 0.0))
                bucket["infer_overheads"].append(res.get("infer_overhead_sec", 0.0))
                bucket["control_overheads"].append(res["control_overhead_sec"])
                bucket["control_latencies_ms"].append(res["avg_control_latency_ms"])
                bucket["n_control_calls"].append(res["n_control_calls"])
                if method == "GARL-ACO":
                    bucket["param_trajectories"].append(res["param_trajectory"])

        # Console Summary
        print("-" * 140)
        print(f"{'Method':17s} | {'Mean Tour':12s} | {'Mean Gap% +/- Std':20s} | {'95% CI Gap':15s} | {'Runtime(s)':15s} | {'Overhead%':10s} | {'Latency(ms)':11s} | {'p-value vs GARL':16s}")
        print("-" * 140)

        garl_gaps = inst_results["GARL-ACO"]["gaps"]

        for method in active_methods:
            d = inst_results[method]
            mean_len = float(np.mean(d["tour_lengths"]))
            mean_gap = float(np.mean(d["gaps"]))
            std_gap = float(np.std(d["gaps"]))
            ci95_gap = 1.96 * std_gap / np.sqrt(len(d["gaps"])) if len(d["gaps"]) > 1 else 0.0
            mean_rt = float(np.mean(d["total_runtimes"]))
            std_rt = float(np.std(d["total_runtimes"]))
            mean_lat = float(np.mean(d["control_latencies_ms"]))

            overhead_pct = float(np.mean([
                (oh / rt * 100.0) if rt > 0 else 0.0
                for oh, rt in zip(d["control_overheads"], d["total_runtimes"])
            ]))

            if method == "GARL-ACO":
                p_str = "Ref (Proposed)"
            else:
                sig = compute_significance(d["gaps"], garl_gaps)
                p_str = f"p={sig['p_value']:.4f}"

            gap_str = f"{mean_gap:6.2f}% +/- {std_gap:5.2f}%"
            ci_str = f"[+/- {ci95_gap:5.2f}%]"
            rt_str = f"{mean_rt:5.2f} +/- {std_rt:.2f}"
            print(f"{method:17s} | {mean_len:12.1f} | {gap_str:20s} | {ci_str:15s} | {rt_str:15s} | {overhead_pct:9.2f}% | {mean_lat:10.2f} | {p_str:16s}")

        print("-" * 140)
        results_summary[inst_name] = inst_results

    # JSON export
    out_path = os.path.join(CHECKPOINT_DIR, output_json)
    json_safe_dict = {}
    for inst, m_dict in results_summary.items():
        json_safe_dict[inst] = {}
        garl_gaps = m_dict["GARL-ACO"]["gaps"]

        for m, data in m_dict.items():
            valid_recs = [r for r in data["recovery_iters"] if r is not None]
            unrec_cnt = len(data["recovery_iters"]) - len(valid_recs)
            std_gap = float(np.std(data["gaps"]))
            ci95_gap = float(1.96 * std_gap / np.sqrt(len(data["gaps"]))) if len(data["gaps"]) > 1 else 0.0

            entry = {
                "tour_lengths": [float(x) for x in data["tour_lengths"]],
                "gaps": [float(x) for x in data["gaps"]],
                "mean_gap": float(np.mean(data["gaps"])),
                "std_gap": std_gap,
                "ci95_gap": ci95_gap,
                "mean_tour": float(np.mean(data["tour_lengths"])),
                "std_tour": float(np.std(data["tour_lengths"])),
                "recovery_iters": [float(x) if x is not None else None for x in data["recovery_iters"]],
                "mean_recovery_iters": float(np.mean(valid_recs)) if len(valid_recs) > 0 else None,
                "unrecovered_count": unrec_cnt,
                "total_runtimes": [float(x) for x in data["total_runtimes"]],
                "mean_runtime_sec": float(np.mean(data["total_runtimes"])),
                "std_runtime_sec": float(np.std(data["total_runtimes"])),
                "mean_aco_runtime_sec": float(np.mean(data["aco_runtimes"])),
                "mean_state_overhead_sec": float(np.mean(data["state_overheads"])),
                "mean_infer_overhead_sec": float(np.mean(data["infer_overheads"])),
                "control_overheads": [float(x) for x in data["control_overheads"]],
                "mean_control_overhead_sec": float(np.mean(data["control_overheads"])),
                "std_control_overhead_sec": float(np.std(data["control_overheads"])),
                "control_latencies_ms": [float(x) for x in data["control_latencies_ms"]],
                "mean_control_latency_ms": float(np.mean(data["control_latencies_ms"])),
                "std_control_latency_ms": float(np.std(data["control_latencies_ms"])),
                "mean_overhead_pct_of_own_runtime": float(np.mean([
                    (oh / rt * 100.0) if rt > 0 else 0.0
                    for oh, rt in zip(data["control_overheads"], data["total_runtimes"])
                ])),
            }
            if m != "GARL-ACO":
                entry["significance_vs_GARL_ACO"] = compute_significance(data["gaps"], garl_gaps)

            json_safe_dict[inst][m] = entry

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe_dict, f, indent=2)

    print(f"\nSaved raw evaluation summary to {out_path}")
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=DEFAULT_N_ITER, help="Iterations per instance")
    parser.add_argument("--n_runs", type=int, default=8, help="Number of random seeds per instance")
    parser.add_argument("--allow_missing_checkpoints", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    args = parser.parse_args()

    evaluate_benchmark(
        n_iter=args.n_iter,
        n_runs=args.n_runs,
        allow_missing_checkpoints=args.allow_missing_checkpoints,
        skip_ablations=args.skip_ablations,
    )
