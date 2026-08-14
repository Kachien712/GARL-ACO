"""
Streamlit Interactive GUI Demo v4 for Adaptive Graph-aware DDPG on Dynamic TSP.
Run with: python -m streamlit run ACO_ML/v4/demo_app.py
"""
import os
import sys
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ACO_ML.v4.config import (
    CHECKPOINT_DIR, BEST_KNOWN_SOLUTIONS, CONTROL_FREQ, FIXED_ACO_PARAMS
)
from ACO_ML.v4.data.tsp_loader import load_dataset_instance
from ACO_ML.v4.data.solutions import get_gap
from ACO_ML.v4.env.dynamic_tsp_env import DynamicTSPEnvironment
from ACO_ML.v4.aco.aco_core import VectorizedACO
from ACO_ML.v4.agent.state_builder import build_state_vector
from ACO_ML.v4.agent.ddpg_agent import DDPGAgent
from ACO_ML.v4.agent.ql_agent import QLearningAgent

st.set_page_config(
    page_title="GARL-ACO Framework for Dynamic TSP",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" GARL-ACO Framework (18 Graph Features)")
st.markdown("### Research Demo — Dynamic Parameter Control with 18 Graph Structural Features")

st.sidebar.header(" Control Panel")

instance_name = st.sidebar.selectbox(
    "Select TSP Instance:",
    options=["kroA200", "pr439", "rat575", "u1060", "pcb1173"],
    index=0
)

method = st.sidebar.radio(
    "Select Dynamic Control Method:",
    options=["Fixed ACO", "ACO + Q-Learning", "GARL-ACO"],
    index=2
)

enable_dynamic_traffic = st.sidebar.toggle(
    "Enable Dynamic Traffic Perturbation (Tắc đường)",
    value=True,
    help="Simulates smooth cyclic congestion scaling on graph edge weights."
)

st.sidebar.markdown("---")
n_iterations = st.sidebar.slider("ACO Iterations:", min_value=50, max_value=300, value=150, step=10)
ant_count = st.sidebar.slider("Number of Ants:", min_value=10, max_value=60, value=30, step=5)

run_button = st.sidebar.button(" Run Live Simulation", type="primary", use_container_width=True)

instance = load_dataset_instance(instance_name)
best_known = BEST_KNOWN_SOLUTIONS[instance_name]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("TSP Instance", instance_name)
with col2:
    st.metric("Cities (Nodes)", instance.n_cities)
with col3:
    st.metric("Best-Known Length", f"{best_known:,}")
with col4:
    st.metric("Traffic Mode", "Dynamic (Congested)" if enable_dynamic_traffic else "Static")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader(" 2D Tour Visualization")
    tour_plot_spot = st.empty()

with right_col:
    st.subheader(" Real-Time Convergence Curve & Gap %")
    conv_plot_spot = st.empty()
    param_plot_spot = st.empty()

if run_button:
    st.toast(f"Starting v4 simulation on {instance_name} with {method}...", icon="🚀")
    
    ddpg_agent = DDPGAgent()
    ddpg_path = os.path.join(CHECKPOINT_DIR, "ddpg_agent.pt")
    if os.path.exists(ddpg_path):
        ddpg_agent.load(ddpg_path)
        
    ql_agent = QLearningAgent()
    ql_path = os.path.join(CHECKPOINT_DIR, "ql_agent.pkl")
    if os.path.exists(ql_path):
        ql_agent.load(ql_path)
        
    dynamic_env = DynamicTSPEnvironment(
        base_dist_matrix=instance.dist_matrix,
        perturb_ratio=0.12,
        perturb_amplitude=0.5,
        perturb_period=20,
        enabled=enable_dynamic_traffic,
        seed=42
    )
    
    aco = VectorizedACO(
        dist_matrix=instance.dist_matrix,
        alpha=FIXED_ACO_PARAMS["alpha"],
        beta=FIXED_ACO_PARAMS["beta"],
        rho=FIXED_ACO_PARAMS["rho"],
        n_ants=ant_count,
        seed=42
    )
    
    convergence_history = []
    param_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for it in range(1, n_iterations + 1):
        active_dist = dynamic_env.step_environment()
        aco.set_dist_matrix(active_dist)
        
        if method == "GARL-ACO" and (it == 1 or it % CONTROL_FREQ == 0):
            graph_stats = dynamic_env.get_change_stats()
            aco_stats = aco.get_search_stats()
            state_vec = build_state_vector(aco_stats, graph_stats, active_dist, current_iter=it, max_iter=n_iterations)
            _, params = ddpg_agent.select_action(state_vec, add_noise=False)
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])
            
        elif method == "ACO + Q-Learning" and (it == 1 or it % CONTROL_FREQ == 0):
            aco_stats = aco.get_search_stats()
            ql_state = ql_agent.discretize_state(aco_stats, current_iter=it, max_iter=n_iterations)
            _, params = ql_agent.select_action(ql_state, evaluate=True)
            aco.set_parameters(params["alpha"], params["beta"], params["rho"])
            
        iter_best = aco.step()
        convergence_history.append(aco.best_length)
        param_history.append((aco.alpha, aco.beta, aco.rho))
        
        if it % 3 == 0 or it == n_iterations:
            progress_bar.progress(it / float(n_iterations))
            curr_gap = get_gap(aco.best_length, instance_name)
            status_text.text(f"Iteration {it}/{n_iterations} | Best Length: {aco.best_length:,.1f} | Gap: {curr_gap:.2f}% | Alpha: {aco.alpha:.2f}, Beta: {aco.beta:.2f}, Rho: {aco.rho:.2f}")
            
            fig_map, ax_map = plt.subplots(figsize=(6, 5))
            coords = instance.coords
            ax_map.scatter(coords[:, 0], coords[:, 1], c='#2c3e50', s=15, zorder=3)
            
            if aco.best_tour is not None:
                tour_idx = np.append(aco.best_tour, aco.best_tour[0])
                ax_map.plot(coords[tour_idx, 0], coords[tour_idx, 1], c='#2980b9', lw=1.2, alpha=0.85, zorder=2)
                
            ax_map.set_title(f"Current Best Tour ({instance_name})", fontsize=11)
            ax_map.axis('off')
            tour_plot_spot.pyplot(fig_map)
            plt.close(fig_map)
            
            fig_conv, ax_conv = plt.subplots(figsize=(6, 3))
            ax_conv.plot(range(1, len(convergence_history) + 1), convergence_history, color='#e74c3c', lw=2)
            ax_conv.axhline(best_known, color='black', linestyle='--', label=f'Best Known ({best_known:,})')
            ax_conv.set_xlabel("ACO Iterations")
            ax_conv.set_ylabel("Tour Length")
            ax_conv.legend(fontsize=9)
            ax_conv.grid(True, linestyle=':', alpha=0.6)
            conv_plot_spot.pyplot(fig_conv)
            plt.close(fig_conv)
            
            if method != "Fixed ACO":
                fig_p, ax_p = plt.subplots(figsize=(6, 2.5))
                params_arr = np.array(param_history)
                ax_p.plot(params_arr[:, 0], label=r'$\alpha$ (Pheromone)', color='#3498db', lw=1.5)
                ax_p.plot(params_arr[:, 1], label=r'$\beta$ (Heuristic)', color='#9b59b6', lw=1.5)
                ax_p.plot(params_arr[:, 2], label=r'$\rho$ (Evaporation)', color='#e67e22', lw=1.5)
                ax_p.set_xlabel("ACO Iterations")
                ax_p.set_ylabel("Parameter Value")
                ax_p.legend(fontsize=8, loc='upper right')
                ax_p.grid(True, linestyle=':', alpha=0.5)
                param_plot_spot.pyplot(fig_p)
                plt.close(fig_p)
                
    st.success(f"Simulation completed! Final Best Length: {aco.best_length:,.1f} (Gap: {get_gap(aco.best_length, instance_name):.2f}%)")
