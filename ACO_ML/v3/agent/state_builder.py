"""
State builder module generating a graph-aware ~27-dimension state representation vector.
Note: gap_to_best_known is strictly excluded from the state space (not available in real-world deployment).
"""
import numpy as np
from ACO_ML.v3.config import ACTION_BOUNDS

STATE_DIM = 27

def build_state_vector(
    aco_stats: dict,
    graph_stats: dict,
    dist_matrix: np.ndarray,
    current_iter: int,
    max_iter: int = 200
) -> np.ndarray:
    """
    Constructs a normalized 27-dimensional feature state vector.
    
    Args:
        aco_stats: Dict returned by VectorizedACO.get_search_stats()
        graph_stats: Dict returned by DynamicTSPEnvironment.get_change_stats()
        dist_matrix: Active symmetric distance matrix (N x N)
        current_iter: Current iteration number
        max_iter: Total expected iterations in episode
        
    Returns:
        state_vec: NumPy array of shape (27,), dtype float32
    """
    N = dist_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    edge_dists = dist_matrix[i_idx, j_idx]
    mean_edge_dist = float(np.mean(edge_dists)) if len(edge_dists) > 0 else 1.0
    
    # Estimate baseline scale: approximate TSP tour scale ~ N * mean_edge_dist * 0.75
    baseline_tour_scale = max(1.0, N * mean_edge_dist * 0.75)
    
    # 1. Edge length distribution histogram (5 bins, normalized)
    hist, _ = np.histogram(edge_dists, bins=5, density=True)
    hist_norm = hist / (np.sum(hist) + 1e-10)
    
    # 2. Graph Dynamic Perturbation Features (6 features)
    p_ratio = graph_stats.get("perturb_ratio", 0.0)
    p_mean_mag = min(2.0, graph_stats.get("mean_change_mag", 0.0))
    p_max_mag = min(5.0, graph_stats.get("max_change_mag", 0.0))
    p_var = min(10.0, graph_stats.get("weight_variance", 0.0) / (mean_edge_dist**2 + 1e-10))
    p_trend = min(2.0, graph_stats.get("volatility_trend", 0.0) / (mean_edge_dist + 1e-10))
    p_steps = min(1.0, graph_stats.get("steps_since_last_change", 0.0) / 50.0)
    
    # 3. Search & Optimization Dynamics (16 features)
    norm_best_len = min(5.0, aco_stats["current_best_length"] / baseline_tour_scale)
    norm_iter_best_len = min(5.0, aco_stats["iter_best_length"] / baseline_tour_scale)
    norm_iter_mean_len = min(5.0, aco_stats["iter_mean_length"] / baseline_tour_scale)
    
    conv_factor = min(2.0, aco_stats["convergence_factor"])
    imp_5 = min(1.0, aco_stats["improvement_rate_5"])
    imp_20 = min(1.0, aco_stats["improvement_rate_20"])
    
    norm_stagnation = min(1.0, aco_stats["stagnation_counter"] / 50.0)
    
    # Pheromone statistics normalized
    tau_mean = aco_stats["tau_mean"]
    tau_norm_mean = min(5.0, tau_mean)
    tau_norm_min = min(5.0, aco_stats["tau_min"] / (tau_mean + 1e-10))
    tau_norm_max = min(10.0, aco_stats["tau_max"] / (tau_mean + 1e-10))
    tau_norm_std = min(5.0, aco_stats["tau_std"] / (tau_mean + 1e-10))
    tau_norm_entropy = min(5.0, aco_stats["tau_entropy"] / np.log(max(2, N * (N - 1) // 2)))
    
    # Current hyperparameter values normalized to [0, 1]
    a_min, a_max = ACTION_BOUNDS["alpha"]
    b_min, b_max = ACTION_BOUNDS["beta"]
    r_min, r_max = ACTION_BOUNDS["rho"]
    
    norm_alpha = (aco_stats["current_alpha"] - a_min) / (a_max - a_min)
    norm_beta = (aco_stats["current_beta"] - b_min) / (b_max - b_min)
    norm_rho = (aco_stats["current_rho"] - r_min) / (r_max - r_min)
    
    # Progress ratio in episode
    progress_ratio = min(1.0, current_iter / float(max_iter))
    
    state_list = [
        # 5 histogram features
        float(hist_norm[0]), float(hist_norm[1]), float(hist_norm[2]), float(hist_norm[3]), float(hist_norm[4]),
        # 6 graph dynamic features
        float(p_ratio), float(p_mean_mag), float(p_max_mag), float(p_var), float(p_trend), float(p_steps),
        # 16 search & parameter features
        float(norm_best_len), float(norm_iter_best_len), float(norm_iter_mean_len),
        float(conv_factor), float(imp_5), float(imp_20), float(norm_stagnation),
        float(tau_norm_mean), float(tau_norm_min), float(tau_norm_max), float(tau_norm_std), float(tau_norm_entropy),
        float(norm_alpha), float(norm_beta), float(norm_rho),
        float(progress_ratio)
    ]
    
    state_vec = np.array(state_list, dtype=np.float32)
    return state_vec
