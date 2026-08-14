"""
Baseline Execution Framework for Fixed ACO, Q-Learning ACO, and DDPG ACO v4.
"""
import os
import sys
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ACO_ML.v4.config import FIXED_ACO_PARAMS, CONTROL_FREQ
from ACO_ML.v4.data.tsp_loader import TSPInstance
from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
from ACO_ML.v4.aco.aco_core import VectorizedACO
from ACO_ML.v4.agent.state_builder import build_state_vector
from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
from ACO_ML.v4.agent.ql_agent import QLearningAgent

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
    
    for it in range(1, n_iter + 1):
        if dynamic_env is not None:
            active_dist = dynamic_env.step_environment()
            aco.set_dist_matrix(active_dist)
            
        iter_best = aco.step()
        convergence_curve.append(aco.best_length)
        iter_best_curve.append(iter_best)
        param_trajectory.append((alpha, beta, rho))
        
    return {
        "method": "Fixed ACO",
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
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
    
    aco_stats = aco.get_search_stats()
    state_tuple = agent.discretize_state(aco_stats, current_iter=0, max_iter=n_iter)
    action_idx, params = agent.select_action(state_tuple, evaluate=evaluate)
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
            aco_stats = aco.get_search_stats()
            state_tuple = agent.discretize_state(aco_stats, current_iter=it, max_iter=n_iter)
            action_idx, params = agent.select_action(state_tuple, evaluate=evaluate)
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])
            
    return {
        "method": "ACO + Q-Learning",
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
    }

def run_ddpg_aco(
    instance: TSPInstance,
    agent: DDPGAgent,
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
        
    convergence_curve = []
    iter_best_curve = []
    param_trajectory = []
    
    aco_stats = aco.get_search_stats()
    state_vec = build_state_vector(aco_stats, graph_stats, instance.dist_matrix, current_iter=0, max_iter=n_iter)
    
    _, params = agent.select_action(state_vec, add_noise=not evaluate)
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
            aco_stats = aco.get_search_stats()
            active_d = dynamic_env.active_dist if dynamic_env is not None else instance.dist_matrix
            state_vec = build_state_vector(aco_stats, graph_stats, active_d, current_iter=it, max_iter=n_iter)
            _, params = agent.select_action(state_vec, add_noise=not evaluate)
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])
            
    return {
        "method": "GARL-ACO",
        "best_length": aco.best_length,
        "best_tour": aco.best_tour,
        "convergence_curve": convergence_curve,
        "iter_best_curve": iter_best_curve,
        "param_trajectory": param_trajectory,
    }
