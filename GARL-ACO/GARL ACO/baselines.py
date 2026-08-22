"""
Baseline Execution Framework v4.3 for Fixed ACO, Q-Learning ACO, DDPG ACO (GARL-ACO),
and Ablation variants (DDPG-Search / NoGraph-DDPG, DDPG-Graph, Discrete-DQN).

Fine-Grained Computational Timing Methodology (VNICT 2026 Camera-Ready Revision):
  Each run measures precise, decomposed wall-clock quantities using time.perf_counter():
    1. aco_runtime_sec (T_ACO): Time spent executing pure ACO stepping and distance updates.
    2. state_overhead_sec (T_state): Time spent building state features (graph and/or search).
    3. infer_overhead_sec (T_infer): Time spent in neural network forward inference.
    4. control_overhead_sec (T_overhead = T_state + T_infer): Total RL control time.
    5. total_runtime_sec: Full wall-clock time for the run.
"""
import os
import sys
import time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from config import FIXED_ACO_PARAMS, CONTROL_FREQ, STATE_DIM, SEARCH_ONLY_STATE_DIM, GRAPH_ONLY_STATE_DIM
    from data.tsp_loader import TSPInstance
    from env.dynamic_tsp_env import DynamicTSPEnvironment
    from aco.aco_core import VectorizedACO
    from agent.state_builder import (
        build_state_vector, build_search_only_state_vector, build_graph_only_state_vector
    )
    from agent.ddpg_agent import DDPGAgent
    from agent.ql_agent import QLearningAgent
    from agent.dqn_agent import DQNAgent
except ImportError:
    from ACO_ML.v4.config import FIXED_ACO_PARAMS, CONTROL_FREQ, STATE_DIM, SEARCH_ONLY_STATE_DIM, GRAPH_ONLY_STATE_DIM
    from ACO_ML.v4.data.tsp_loader import TSPInstance
    from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
    from ACO_ML.v4.aco.aco_core import VectorizedACO
    from ACO_ML.v4.agent.state_builder import (
        build_state_vector, build_search_only_state_vector, build_graph_only_state_vector
    )
    from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
    from ACO_ML.v4.agent.ql_agent import QLearningAgent
    from ACO_ML.v4.agent.dqn_agent import DQNAgent

def _warmup_torch_actor(actor_module: torch.nn.Module, state_dim: int):
    """Runs one throwaway forward pass to eliminate first-call PyTorch lazy initialization."""
    dummy = torch.zeros((1, state_dim), dtype=torch.float32)
    actor_module.eval()
    with torch.no_grad():
        _ = actor_module(dummy)
    actor_module.train()

def run_fixed_aco(
    instance: TSPInstance,
    n_iter: int = 200,
    dynamic_env: DynamicTSPEnvironment = None,
    alpha: float = FIXED_ACO_PARAMS["alpha"],
    beta: float = FIXED_ACO_PARAMS["beta"],
    rho: float = FIXED_ACO_PARAMS["rho"],
    seed: int = 42
) -> dict:
    aco = VectorizedACO(
        dist_matrix=instance.dist_matrix,
        alpha=alpha,
        beta=beta,
        rho=rho,
        seed=seed
    )

    if dynamic_env is not None:
        dynamic_env.reset()

    convergence_curve = []
    iter_best_curve = []
    param_trajectory = []

    t_run_start = time.perf_counter()

    for it in range(1, n_iter + 1):
        if dynamic_env is not None:
            active_dist = dynamic_env.step_environment()
            aco.set_dist_matrix(active_dist)

        iter_best = aco.step()
        convergence_curve.append(aco.best_length)
        iter_best_curve.append(iter_best)
        param_trajectory.append((alpha, beta, rho))

    total_runtime_sec = time.perf_counter() - t_run_start

    return {
        "method": "Fixed ACO",
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
        "total_runtime_sec": total_runtime_sec,
        "aco_runtime_sec": total_runtime_sec,
        "state_overhead_sec": 0.0,
        "infer_overhead_sec": 0.0,
        "control_overhead_sec": 0.0,
        "avg_control_latency_ms": 0.0,
        "n_control_calls": 0,
    }

def run_ql_aco(
    instance: TSPInstance,
    agent: QLearningAgent,
    n_iter: int = 200,
    dynamic_env: DynamicTSPEnvironment = None,
    control_freq: int = CONTROL_FREQ,
    evaluate: bool = True,
    seed: int = 42
) -> dict:
    aco = VectorizedACO(
        dist_matrix=instance.dist_matrix,
        alpha=1.0, beta=5.0, rho=0.1,
        seed=seed
    )

    if dynamic_env is not None:
        dynamic_env.reset()

    convergence_curve = []
    iter_best_curve = []
    param_trajectory = []
    
    t_state_total = 0.0
    t_infer_total = 0.0
    n_control_calls = 0

    t_run_start = time.perf_counter()

    # Step 0 Initial Decision
    t0 = time.perf_counter()
    aco_stats = aco.get_search_stats()
    state_tuple = agent.discretize_state(aco_stats, current_iter=0, max_iter=n_iter)
    t_state_total += time.perf_counter() - t0

    t0 = time.perf_counter()
    action_idx, params = agent.select_action(state_tuple, evaluate=evaluate)
    t_infer_total += time.perf_counter() - t0
    n_control_calls += 1
    aco.set_parameters(params["alpha"], params["beta"], params["rho"])

    for it in range(1, n_iter + 1):
        if dynamic_env is not None:
            active_dist = dynamic_env.step_environment()
            aco.set_dist_matrix(active_dist)

        iter_best = aco.step()
        convergence_curve.append(aco.best_length)
        iter_best_curve.append(iter_best)
        param_trajectory.append((aco.alpha, aco.beta, aco.rho))

        if it % control_freq == 0:
            t0 = time.perf_counter()
            aco_stats = aco.get_search_stats()
            state_tuple = agent.discretize_state(aco_stats, current_iter=it, max_iter=n_iter)
            t_state_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            action_idx, params = agent.select_action(state_tuple, evaluate=evaluate)
            t_infer_total += time.perf_counter() - t0
            n_control_calls += 1
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])

    total_runtime_sec = time.perf_counter() - t_run_start
    control_time_total = t_state_total + t_infer_total
    aco_runtime_sec = max(0.0, total_runtime_sec - control_time_total)
    avg_latency_ms = (control_time_total / n_control_calls * 1000.0) if n_control_calls > 0 else 0.0

    return {
        "method": "ACO + Q-Learning",
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
        "total_runtime_sec": total_runtime_sec,
        "aco_runtime_sec": aco_runtime_sec,
        "state_overhead_sec": t_state_total,
        "infer_overhead_sec": t_infer_total,
        "control_overhead_sec": control_time_total,
        "avg_control_latency_ms": avg_latency_ms,
        "n_control_calls": n_control_calls,
    }

def run_ddpg_aco(
    instance: TSPInstance,
    agent: DDPGAgent,
    n_iter: int = 200,
    dynamic_env: DynamicTSPEnvironment = None,
    control_freq: int = CONTROL_FREQ,
    evaluate: bool = True,
    seed: int = 42,
    state_mode: str = "full",  # "full" (34D) | "search_only" (16D) | "graph_only" (18D)
) -> dict:
    if state_mode not in ("full", "search_only", "graph_only"):
        raise ValueError(f"state_mode must be 'full', 'search_only' or 'graph_only', got {state_mode!r}")

    aco = VectorizedACO(
        dist_matrix=instance.dist_matrix,
        alpha=1.0, beta=5.0, rho=0.1,
        seed=seed
    )

    if dynamic_env is not None:
        dynamic_env.reset()
        graph_stats = dynamic_env.get_change_stats()
    else:
        graph_stats = {
            "perturb_ratio": 0.0, "mean_change_mag": 0.0, "max_change_mag": 0.0,
            "weight_variance": float(np.var(instance.dist_matrix)), "volatility_trend": 0.0,
            "steps_since_last_change": 0.0
        }

    def _build_state(aco_stats, graph_stats, dist_mat, cur_it):
        if state_mode == "full":
            return build_state_vector(aco_stats, graph_stats, dist_mat, current_iter=cur_it, max_iter=n_iter)
        elif state_mode == "search_only":
            return build_search_only_state_vector(aco_stats, dist_mat, current_iter=cur_it, max_iter=n_iter)
        else:
            return build_graph_only_state_vector(graph_stats, dist_mat)

    state_dims = {"full": STATE_DIM, "search_only": SEARCH_ONLY_STATE_DIM, "graph_only": GRAPH_ONLY_STATE_DIM}
    _warmup_torch_actor(agent.actor, state_dims[state_mode])

    convergence_curve = []
    iter_best_curve = []
    param_trajectory = []
    
    t_state_total = 0.0
    t_infer_total = 0.0
    n_control_calls = 0

    t_run_start = time.perf_counter()

    # Step 0 Initial Decision
    t0 = time.perf_counter()
    aco_stats = aco.get_search_stats()
    state_vec = _build_state(aco_stats, graph_stats, instance.dist_matrix, 0)
    t_state_total += time.perf_counter() - t0

    t0 = time.perf_counter()
    _, params = agent.select_action(state_vec, add_noise=not evaluate)
    t_infer_total += time.perf_counter() - t0
    n_control_calls += 1
    aco.set_parameters(params["alpha"], params["beta"], params["rho"])

    for it in range(1, n_iter + 1):
        if dynamic_env is not None:
            active_dist = dynamic_env.step_environment()
            aco.set_dist_matrix(active_dist)
            graph_stats = dynamic_env.get_change_stats()

        iter_best = aco.step()
        convergence_curve.append(aco.best_length)
        iter_best_curve.append(iter_best)
        param_trajectory.append((aco.alpha, aco.beta, aco.rho))

        if it % control_freq == 0:
            t0 = time.perf_counter()
            aco_stats = aco.get_search_stats()
            active_d = dynamic_env.active_dist if dynamic_env is not None else instance.dist_matrix
            state_vec = _build_state(aco_stats, graph_stats, active_d, it)
            t_state_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, params = agent.select_action(state_vec, add_noise=not evaluate)
            t_infer_total += time.perf_counter() - t0
            n_control_calls += 1
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])

    total_runtime_sec = time.perf_counter() - t_run_start
    control_time_total = t_state_total + t_infer_total
    aco_runtime_sec = max(0.0, total_runtime_sec - control_time_total)
    avg_latency_ms = (control_time_total / n_control_calls * 1000.0) if n_control_calls > 0 else 0.0

    method_name = "GARL-ACO" if state_mode == "full" else ("DDPG-Search" if state_mode == "search_only" else "DDPG-Graph")

    return {
        "method": method_name,
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
        "total_runtime_sec": total_runtime_sec,
        "aco_runtime_sec": aco_runtime_sec,
        "state_overhead_sec": t_state_total,
        "infer_overhead_sec": t_infer_total,
        "control_overhead_sec": control_time_total,
        "avg_control_latency_ms": avg_latency_ms,
        "n_control_calls": n_control_calls,
    }

def run_dqn_aco(
    instance: TSPInstance,
    agent: DQNAgent,
    n_iter: int = 200,
    dynamic_env: DynamicTSPEnvironment = None,
    control_freq: int = CONTROL_FREQ,
    evaluate: bool = True,
    seed: int = 42
) -> dict:
    aco = VectorizedACO(
        dist_matrix=instance.dist_matrix,
        alpha=1.0, beta=5.0, rho=0.1,
        seed=seed
    )

    if dynamic_env is not None:
        dynamic_env.reset()
        graph_stats = dynamic_env.get_change_stats()
    else:
        graph_stats = {
            "perturb_ratio": 0.0, "mean_change_mag": 0.0, "max_change_mag": 0.0,
            "weight_variance": float(np.var(instance.dist_matrix)), "volatility_trend": 0.0,
            "steps_since_last_change": 0.0
        }

    _warmup_torch_actor(agent.q_net, STATE_DIM)

    convergence_curve = []
    iter_best_curve = []
    param_trajectory = []
    
    t_state_total = 0.0
    t_infer_total = 0.0
    n_control_calls = 0

    t_run_start = time.perf_counter()

    # Step 0 Initial Decision
    t0 = time.perf_counter()
    aco_stats = aco.get_search_stats()
    state_vec = build_state_vector(aco_stats, graph_stats, instance.dist_matrix, current_iter=0, max_iter=n_iter)
    t_state_total += time.perf_counter() - t0

    t0 = time.perf_counter()
    _, params = agent.select_action(state_vec, evaluate=evaluate)
    t_infer_total += time.perf_counter() - t0
    n_control_calls += 1
    aco.set_parameters(params["alpha"], params["beta"], params["rho"])

    for it in range(1, n_iter + 1):
        if dynamic_env is not None:
            active_dist = dynamic_env.step_environment()
            aco.set_dist_matrix(active_dist)
            graph_stats = dynamic_env.get_change_stats()

        iter_best = aco.step()
        convergence_curve.append(aco.best_length)
        iter_best_curve.append(iter_best)
        param_trajectory.append((aco.alpha, aco.beta, aco.rho))

        if it % control_freq == 0:
            t0 = time.perf_counter()
            aco_stats = aco.get_search_stats()
            active_d = dynamic_env.active_dist if dynamic_env is not None else instance.dist_matrix
            state_vec = build_state_vector(aco_stats, graph_stats, active_d, current_iter=it, max_iter=n_iter)
            t_state_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, params = agent.select_action(state_vec, evaluate=evaluate)
            t_infer_total += time.perf_counter() - t0
            n_control_calls += 1
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])

    total_runtime_sec = time.perf_counter() - t_run_start
    control_time_total = t_state_total + t_infer_total
    aco_runtime_sec = max(0.0, total_runtime_sec - control_time_total)
    avg_latency_ms = (control_time_total / n_control_calls * 1000.0) if n_control_calls > 0 else 0.0

    return {
        "method": "Discrete-DQN",
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
        "total_runtime_sec": total_runtime_sec,
        "aco_runtime_sec": aco_runtime_sec,
        "state_overhead_sec": t_state_total,
        "infer_overhead_sec": t_infer_total,
        "control_overhead_sec": control_time_total,
        "avg_control_latency_ms": avg_latency_ms,
        "n_control_calls": n_control_calls,
    }
