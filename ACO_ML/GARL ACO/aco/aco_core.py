"""
Vectorized High-Performance ACO Engine for v4 architecture.
"""
import numpy as np

class VectorizedACO:
    def __init__(
        self,
        dist_matrix: np.ndarray,
        alpha: float = 1.0,
        beta: float = 5.0,
        rho: float = 0.1,
        n_ants: int = None,
        q_constant: float = 100.0,
        seed: int = None
    ):
        self.dist_matrix = dist_matrix.copy()
        self.N = self.dist_matrix.shape[0]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q_constant = float(q_constant)
        
        if n_ants is None:
            self.n_ants = min(50, max(10, self.N // 4))
        else:
            self.n_ants = int(n_ants)
            
        self.rng = np.random.RandomState(seed)
        
        eps = 1e-10
        self.eta = 1.0 / (self.dist_matrix + eps)
        np.fill_diagonal(self.eta, 0.0)
        
        self.tau = np.ones((self.N, self.N), dtype=np.float64)
        np.fill_diagonal(self.tau, 0.0)
        
        self.best_tour = None
        self.best_length = float('inf')
        self.iter_best_length = float('inf')
        self.iter_lengths = np.array([])
        self.stagnation_counter = 0
        self.history_best_lengths = []
        self.iteration = 0

    def set_parameters(self, alpha: float, beta: float, rho: float):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)

    def set_dist_matrix(self, new_dist_matrix: np.ndarray):
        self.dist_matrix = new_dist_matrix.copy()
        eps = 1e-10
        self.eta = 1.0 / (self.dist_matrix + eps)
        np.fill_diagonal(self.eta, 0.0)

    def step(self) -> float:
        self.iteration += 1
        
        tau_alpha = np.power(np.maximum(self.tau, 1e-10), self.alpha)
        eta_beta = np.power(self.eta, self.beta)
        desirability = tau_alpha * eta_beta
        np.fill_diagonal(desirability, 0.0)
        
        tours = np.zeros((self.n_ants, self.N), dtype=np.int32)
        visited = np.zeros((self.n_ants, self.N), dtype=bool)
        
        start_cities = self.rng.choice(self.N, size=self.n_ants, replace=True)
        tours[:, 0] = start_cities
        
        ant_indices = np.arange(self.n_ants)
        visited[ant_indices, start_cities] = True
        current_cities = start_cities.copy()
        
        for step_idx in range(1, self.N):
            step_desire = desirability[current_cities, :].copy()
            step_desire[visited] = 0.0
            
            row_sums = step_desire.sum(axis=1, keepdims=True)
            zero_mask = (row_sums.squeeze() <= 1e-12)
            if np.any(zero_mask):
                fallback_prob = (~visited[zero_mask]).astype(np.float64)
                fallback_sums = fallback_prob.sum(axis=1, keepdims=True)
                fallback_sums[fallback_sums == 0] = 1.0
                step_desire[zero_mask] = fallback_prob / fallback_sums
                row_sums[zero_mask] = 1.0
                
            probs = step_desire / row_sums
            cum_probs = np.cumsum(probs, axis=1)
            r = self.rng.rand(self.n_ants, 1)
            
            next_cities = np.argmax(cum_probs > r, axis=1)
            
            tours[:, step_idx] = next_cities
            visited[ant_indices, next_cities] = True
            current_cities = next_cities
            
        u_cities = tours
        v_cities = np.roll(tours, -1, axis=1)
        
        edge_dists = self.dist_matrix[u_cities, v_cities]
        self.iter_lengths = edge_dists.sum(axis=1)
        
        iter_min_idx = np.argmin(self.iter_lengths)
        self.iter_best_length = float(self.iter_lengths[iter_min_idx])
        iter_best_tour = tours[iter_min_idx].copy()
        
        if self.iter_best_length < self.best_length:
            self.best_length = self.iter_best_length
            self.best_tour = iter_best_tour.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        self.history_best_lengths.append(self.best_length)
        
        self.tau *= (1.0 - self.rho)
        deposits = self.q_constant / (self.iter_lengths + 1e-10)
        
        for k in range(self.n_ants):
            ant_tour = tours[k]
            ant_u = ant_tour
            ant_v = np.roll(ant_tour, -1)
            dep = deposits[k]
            
            np.add.at(self.tau, (ant_u, ant_v), dep)
            np.add.at(self.tau, (ant_v, ant_u), dep)
            
        if self.best_tour is not None:
            best_u = self.best_tour
            best_v = np.roll(self.best_tour, -1)
            best_dep = self.q_constant / (self.best_length + 1e-10)
            np.add.at(self.tau, (best_u, best_v), best_dep)
            np.add.at(self.tau, (best_v, best_u), best_dep)
            
        np.clip(self.tau, a_min=1e-5, a_max=1e5, out=self.tau)
        np.fill_diagonal(self.tau, 0.0)
        
        return self.iter_best_length

    def get_search_stats(self) -> dict:
        i_idx, j_idx = np.triu_indices(self.N, k=1)
        tau_values = self.tau[i_idx, j_idx]
        
        tau_min = float(np.min(tau_values))
        tau_max = float(np.max(tau_values))
        tau_mean = float(np.mean(tau_values))
        tau_std = float(np.std(tau_values))
        
        tau_norm = tau_values / (np.sum(tau_values) + 1e-10)
        entropy = -float(np.sum(tau_norm * np.log(tau_norm + 1e-12)))
        
        iter_std = float(np.std(self.iter_lengths)) if len(self.iter_lengths) > 0 else 0.0
        iter_mean = float(np.mean(self.iter_lengths)) if len(self.iter_lengths) > 0 else 1.0
        
        hist_len = len(self.history_best_lengths)
        if hist_len >= 6:
            imp_5 = (self.history_best_lengths[-6] - self.history_best_lengths[-1]) / (self.history_best_lengths[-6] + 1e-10)
        else:
            imp_5 = 0.0
            
        if hist_len >= 21:
            imp_20 = (self.history_best_lengths[-21] - self.history_best_lengths[-1]) / (self.history_best_lengths[-21] + 1e-10)
        else:
            imp_20 = imp_5
            
        return {
            "current_best_length": self.best_length,
            "iter_best_length": self.iter_best_length,
            "iter_mean_length": iter_mean,
            "iter_std_length": iter_std,
            "convergence_factor": iter_std / (iter_mean + 1e-10),
            "improvement_rate_5": max(0.0, float(imp_5)),
            "improvement_rate_20": max(0.0, float(imp_20)),
            "stagnation_counter": self.stagnation_counter,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "tau_mean": tau_mean,
            "tau_std": tau_std,
            "tau_entropy": entropy,
            "current_alpha": self.alpha,
            "current_beta": self.beta,
            "current_rho": self.rho,
        }
