"""
TSPLIB dataset parser and distance matrix builder.
"""
import os
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from ACO_ML.v4.config import DATASET_DIR

class TSPInstance:
    def __init__(self, name: str, coords: np.ndarray, dist_matrix: np.ndarray, graph: nx.Graph):
        self.name = name
        self.coords = coords                # Shape (N, 2)
        self.dist_matrix = dist_matrix      # Shape (N, N)
        self.graph = graph                  # NetworkX graph representation
        self.n_cities = len(coords)

def parse_tsp_file(filepath: str, round_int: bool = True) -> TSPInstance:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TSP file not found at {filepath}")
        
    instance_name = os.path.basename(filepath).replace(".tsp", "")
    coords_list = []
    in_coord_section = False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue
            if line_str.startswith("EOF") or line_str.startswith("DEMAND_SECTION"):
                break
                
            if in_coord_section:
                parts = line_str.split()
                if len(parts) >= 3:
                    try:
                        x, y = float(parts[1]), float(parts[2])
                        coords_list.append([x, y])
                    except ValueError:
                        continue
                        
    coords = np.array(coords_list, dtype=np.float64)
    if len(coords) == 0:
        raise ValueError(f"Failed to parse coordinates from {filepath}")
        
    raw_dist = cdist(coords, coords, metric='euclidean')
    if round_int:
        dist_matrix = np.round(raw_dist).astype(np.float64)
    else:
        dist_matrix = raw_dist.astype(np.float64)
        
    np.fill_diagonal(dist_matrix, 0.0)
    
    graph = nx.from_numpy_array(dist_matrix)
    nx.set_node_attributes(graph, {i: coords[i] for i in range(len(coords))}, 'pos')
    
    return TSPInstance(name=instance_name, coords=coords, dist_matrix=dist_matrix, graph=graph)

def load_dataset_instance(name_or_path: str, dataset_dir: str = DATASET_DIR) -> TSPInstance:
    if os.path.exists(name_or_path):
        return parse_tsp_file(name_or_path)
    
    clean_name = os.path.basename(name_or_path).replace(".tsp", "")
    full_path = os.path.join(dataset_dir, f"{clean_name}.tsp")
    return parse_tsp_file(full_path)
