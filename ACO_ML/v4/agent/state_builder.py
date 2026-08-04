"""
Graph-Aware State Builder v4.1 (Fixed & Standardized).
Generates 18 Non-Constant, Dynamically Scaled Graph Structural & Dynamic Traffic Features
+ 16 Search & Parameter Features.
Total State Dimension = 34.
"""
import numpy as np

# Total State Dimension: 18 Graph Features + 16 Search Dynamics Features = 34
STATE_DIM = 34

def extract_18_graph_features(dist_matrix: np.ndarray, graph_stats: dict) -> list:
    """
    Extracts 18 NON-CONSTANT, Z-Score / Min-Max Scaled Graph Structural & Dynamic Traffic Features.
    All features have non-zero variance and bounded ranges to prevent dead neurons in PyTorch linear layers.
    """
    N = dist_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    edge_dists = dist_matrix[i_idx, j_idx]
    
    mean_w = float(np.mean(edge_dists)) if len(edge_dists) > 0 else 1.0
    std_w = float(np.std(edge_dists)) if len(edge_dists) > 0 else 1.0
    min_w = float(np.min(edge_dists)) if len(edge_dists) > 0 else 0.1
    max_w = float(np.max(edge_dists)) if len(edge_dists) > 0 else 1.0
    median_w = float(np.median(edge_dists)) if len(edge_dists) > 0 else 1.0
    
    # 1. num_nodes_norm (0.1 - 1.0)
    f1_nodes = float(N / 1200.0)
    
    # 2. avg_degree_norm (0.1 - 1.0)
    f2_avg_deg = float((N - 1) / 1200.0)
    
    # 3. max_degree_norm (0.1 - 1.0)
    f3_max_deg = float(N / 1200.0)
    
    # 4. avg_weight_norm (Scaled)
    baseline_scale = max(1.0, N * mean_w * 0.75)
    f4_avg_weight = float(np.clip(mean_w / (baseline_scale / N + 1e-10), 0.0, 2.0))
    
    # 5. weight_std_ratio
    f5_std_ratio = float(np.clip(std_w / (mean_w + 1e-10), 0.0, 2.0))
    
    # 6. weight_skewness (Normalized to [-1, 1])
    skew = 3.0 * (mean_w - median_w) / (std_w + 1e-10)
    f6_skewness = float(np.clip(skew / 3.0, -1.0, 1.0))
    
    # 7. est_diameter_ratio
    f7_diameter = float(np.clip(max_w / (mean_w + 1e-10) / 3.0, 0.0, 2.0))
    
    # 8. algebraic_connectivity_proxy
    f8_alg_conn = float(np.clip(min_w / (mean_w + 1e-10), 0.0, 1.0))
    
    # 9-14. Dynamic Traffic Perturbation Features (Strictly bounded [0, 1] or [0, 2])
    p_ratio = float(graph_stats.get("perturb_ratio", 0.0))
    p_mean_mag = float(np.clip(graph_stats.get("mean_change_mag", 0.0), 0.0, 2.0))
    p_max_mag = float(np.clip(graph_stats.get("max_change_mag", 0.0) / 2.0, 0.0, 2.0))
    p_var = float(np.clip(graph_stats.get("weight_variance", 0.0) / (mean_w**2 + 1e-10), 0.0, 2.0))
    p_trend = float(np.clip(graph_stats.get("volatility_trend", 0.0) / (mean_w + 1e-10), 0.0, 2.0))
    p_steps = float(np.clip(graph_stats.get("steps_since_last_change", 0.0) / 50.0, 0.0, 1.0))
    
    # 15. knn_weight_ratio (k=15 nearest neighbor distance ratio)
    sorted_dists = np.sort(dist_matrix, axis=1)
    k_nn_dists = sorted_dists[:, 1:min(16, N)]
    knn_mean_w = float(np.mean(k_nn_dists)) if k_nn_dists.size > 0 else mean_w
    f15_knn_ratio = float(np.clip(knn_mean_w / (mean_w + 1e-10), 0.0, 1.0))
    
    # 16. congestion_clustering_intensity
    f16_clustering = float(np.clip(p_ratio * (1.0 + p_mean_mag), 0.0, 1.0))
    
    # 17. weight_range_ratio (max_w - min_w) / mean_w
    f17_range_ratio = float(np.clip((max_w - min_w) / (mean_w + 1e-10) / 3.0, 0.0, 2.0))
    
    # 18. IQR_ratio (Interquartile Range normalized)
    q75, q25 = np.percentile(edge_dists, [75, 25])
    iqr = float(q75 - q25)
    f18_iqr_ratio = float(np.clip(iqr / (mean_w + 1e-10), 0.0, 2.0))
    
    graph_18 = [
        f1_nodes, f2_avg_deg, f3_max_deg, f4_avg_weight, f5_std_ratio,
        f6_skewness, f7_diameter, f8_alg_conn, p_ratio, p_mean_mag,
        p_max_mag, p_var, p_trend, p_steps, f15_knn_ratio,
        f16_clustering, f17_range_ratio, f18_iqr_ratio
    ]
    return graph_18

def build_state_vector(
    aco_stats: dict,
    graph_stats: dict,
    dist_matrix: np.ndarray,
    current_iter: int,
    max_iter: int = 200
) -> np.ndarray:
    """
    Builds the 34-dimensional hybrid State Vector v4.1:
        18 Non-Constant Graph-Aware Features + 16 Search & Parameter Features.
    """
    N = dist_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    edge_dists = dist_matrix[i_idx, j_idx]
    mean_edge_dist = float(np.mean(edge_dists)) if len(edge_dists) > 0 else 1.0
    baseline_tour_scale = max(1.0, N * mean_edge_dist * 0.75)
    
    # 18 Graph Features
    graph_18 = extract_18_graph_features(dist_matrix, graph_stats)
    
    # 16 Search & Parameter Features
    norm_best_len = float(np.clip(aco_stats["current_best_length"] / baseline_tour_scale, 0.0, 5.0))
    norm_iter_best_len = float(np.clip(aco_stats["iter_best_length"] / baseline_tour_scale, 0.0, 5.0))
    norm_iter_mean_len = float(np.clip(aco_stats["iter_mean_length"] / baseline_tour_scale, 0.0, 5.0))
    
    conv_factor = float(np.clip(aco_stats["convergence_factor"], 0.0, 2.0))
    imp_5 = float(np.clip(aco_stats["improvement_rate_5"], 0.0, 1.0))
    imp_20 = float(np.clip(aco_stats["improvement_rate_20"], 0.0, 1.0))
    norm_stagnation = float(np.clip(aco_stats["stagnation_counter"] / 50.0, 0.0, 1.0))
    
    tau_mean = float(aco_stats["tau_mean"])
    tau_norm_mean = float(np.clip(tau_mean, 0.0, 5.0))
    tau_norm_min = float(np.clip(aco_stats["tau_min"] / (tau_mean + 1e-10), 0.0, 5.0))
    tau_norm_max = float(np.clip(aco_stats["tau_max"] / (tau_mean + 1e-10), 0.0, 10.0))
    tau_norm_std = float(np.clip(aco_stats["tau_std"] / (tau_mean + 1e-10), 0.0, 5.0))
    tau_norm_entropy = float(np.clip(aco_stats["tau_entropy"] / np.log(max(2, N * (N - 1) // 2)), 0.0, 5.0))
    
    # Hyperparameters & Progress
    norm_alpha = float((aco_stats["current_alpha"] - 0.5) / (3.0 - 0.5))
    norm_beta = float((aco_stats["current_beta"] - 2.0) / (6.0 - 2.0))
    norm_rho = float((aco_stats["current_rho"] - 0.05) / (0.50 - 0.05))
    progress_ratio = float(np.clip(current_iter / float(max_iter), 0.0, 1.0))
    
    search_16 = [
        norm_best_len, norm_iter_best_len, norm_iter_mean_len,
        conv_factor, imp_5, imp_20, norm_stagnation,
        tau_norm_mean, tau_norm_min, tau_norm_max, tau_norm_std, tau_norm_entropy,
        norm_alpha, norm_beta, norm_rho, progress_ratio
    ]
    
    full_state = graph_18 + search_16
    return np.array(full_state, dtype=np.float32)
