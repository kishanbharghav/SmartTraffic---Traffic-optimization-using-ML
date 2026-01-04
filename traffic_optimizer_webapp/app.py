import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import random
from pathlib import Path

from src.simulation import IntersectionSim, optimize_signals, prepare_instance_features

st.set_page_config(page_title="Traffic Flow Optimizer", layout="wide")
st.title("🚦 Traffic Flow Optimizer: Comparing ML Models")

# --- VISUALIZATION FUNCTION ---
def draw_intersections(intersections, title):
    num_intersections = len(intersections)
    fig, axes = plt.subplots(1, num_intersections, figsize=(num_intersections * 4, 3.5))
    if num_intersections == 1: axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for ax, inter in zip(axes, intersections):
        ax.add_patch(plt.Rectangle((0, 0.45), inter.road_length, 0.1, color='#7D7D7D'))
        ax.axvline(x=1, color='white', linestyle='--', linewidth=2)
        for vehicle in inter.vehicles:
            ax.add_patch(plt.Rectangle((vehicle.position, 0.46), 4, 0.08, color=vehicle.color, ec='black', lw=0.5))
        light_color = 'green' if inter.is_green else 'red'
        ax.add_patch(plt.Circle((1, 0.7), 0.05, color=light_color, ec='black'))
        ax.text(inter.road_length / 2, 0.8, f"Green Time: {inter.green_time:.0f}s", ha='center', color='white', weight='bold')
        if inter.incident_active:
            ax.text(inter.road_length / 2, 0.1, "💥 INCIDENT 💥", ha='center', color='yellow', weight='bold', fontsize=12)
        ax.set_title(f"Intersection {inter.intersection_id}")
        ax.set_xlim(-5, inter.road_length + 5)
        ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xticks([]); ax.spines[:].set_visible(False)
        ax.patch.set_alpha(0.0)

    fig.patch.set_alpha(0.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# --- MAIN APP LOGIC ---

# Sidebar controls
st.sidebar.title("Simulation Controls")

MODEL_FILES = {
    "Random Forest": "models/Random_Forest_pipeline.pkl",
    "Gradient Boosting": "models/Gradient_Boosting_pipeline.pkl",
    "Ridge Regression": "models/Ridge_Regression_pipeline.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    options=list(MODEL_FILES.keys())
)

try:
    model_path = MODEL_FILES[selected_model_name]
    model = joblib.load(model_path)
    df = pd.read_csv("data/synthetic_traffic_dataset_6000.csv", parse_dates=["timestamp"])
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e.filename}. Please run the trainer script first.")
    st.stop()

n_intersections = st.sidebar.slider("Number of intersections", 1, 4, 2)
base_cycle = st.sidebar.slider("Base cycle time (seconds)", 60, 180, 120)
steps = st.sidebar.slider("Number of simulation steps", 10, 200, 75)
run_simulation = st.sidebar.button("▶️ Run Simulation")
st.sidebar.title("Event Controls")
if st.sidebar.button("💥 Trigger Random Incident"):
    st.session_state.incident_to_trigger = True

if 'running' not in st.session_state: st.session_state.running = False
if 'incident_to_trigger' not in st.session_state: st.session_state.incident_to_trigger = False

if run_simulation:
    st.session_state.running = True
    all_ids = df['intersection_id'].unique()
    ids_to_simulate = all_ids[:n_intersections]
    st.session_state.ml_intersections = [IntersectionSim(i, df[df['intersection_id']==i]) for i in ids_to_simulate]
    st.session_state.ft_intersections = [IntersectionSim(i, df[df['intersection_id']==i]) for i in ids_to_simulate]
    st.session_state.history = []
    st.session_state.step = 0

if not st.session_state.running:
    st.info("Choose a model and press ▶️ Run Simulation in the sidebar to start.")
else:
    st.header("Live Simulation View")
    col1, col2 = st.columns(2)
    with col1: ml_plot_placeholder = st.empty()
    with col2: ft_plot_placeholder = st.empty()
    st.header("Performance Metrics")
    metrics_placeholder = st.empty()
    st.header("Performance Over Time")
    chart_placeholder = st.empty()
    progress = st.progress(0.0)

    for step in range(st.session_state.step, steps):
        st.session_state.step = step
        if st.session_state.incident_to_trigger:
            rand_index = random.randint(0, len(st.session_state.ml_intersections) - 1)
            st.session_state.ml_intersections[rand_index].trigger_incident()
            st.session_state.ft_intersections[rand_index].trigger_incident()
            st.toast(f"💥 Incident triggered at Intersection {st.session_state.ml_intersections[rand_index].intersection_id}!")
            st.session_state.incident_to_trigger = False

        feats = [prepare_instance_features(inter) for inter in st.session_state.ml_intersections]
        preds = model.predict(pd.DataFrame(feats))
        ml_green_times = optimize_signals(preds, base_cycle=base_cycle)
        for i, inter in enumerate(st.session_state.ml_intersections):
            inter.apply_control(predicted_flow=float(preds[i]), green_time=int(ml_green_times[i]))

        for inter in st.session_state.ft_intersections:
            inter.apply_control(predicted_flow=inter.predicted_flow, green_time=60)
        
        ml_state = pd.DataFrame([inter.get_state() for inter in st.session_state.ml_intersections])
        ft_state = pd.DataFrame([inter.get_state() for inter in st.session_state.ft_intersections])
        
        # Using simplified, stable column names for history
        st.session_state.history.append({
            "step": step,
            "ML Mean Queue": ml_state['queue_length'].mean(),
            "Fixed-Time Mean Queue": ft_state['queue_length'].mean(),
            "ML Mean Wait": ml_state['avg_wait_time'].mean(),
            "Fixed-Time Mean Wait": ft_state['avg_wait_time'].mean()
        })

        ml_fig = draw_intersections(st.session_state.ml_intersections, f"Smart ({selected_model_name})")
        ml_plot_placeholder.pyplot(ml_fig)
        plt.close(ml_fig)
        
        ft_fig = draw_intersections(st.session_state.ft_intersections, "Standard (Fixed-Time)")
        ft_plot_placeholder.pyplot(ft_fig)
        plt.close(ft_fig)

        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Smart ({selected_model_name})")
                st.dataframe(ml_state[['intersection_id', 'queue_length', 'avg_wait_time', 'green_time']].set_index('intersection_id'))
            with col2:
                st.subheader("Standard (Fixed-Time)")
                st.dataframe(ft_state[['intersection_id', 'queue_length', 'avg_wait_time', 'green_time']].set_index('intersection_id'))
        
        # ✅ FIXED: This section now correctly renames columns for the chart legend, ensuring both lines are always displayed.
        with chart_placeholder.container():
            history_df = pd.DataFrame(st.session_state.history).set_index('step')
            # Create a copy to rename for charting, ensuring clean legends
            chart_data = history_df.rename(columns={
                "ML Mean Queue": f"{selected_model_name} Mean Queue",
                "ML Mean Wait": f"{selected_model_name} Mean Wait"
            })

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Mean Queue Length Over Time")
                # Plot the renamed columns
                st.line_chart(chart_data[[f"{selected_model_name} Mean Queue", "Fixed-Time Mean Queue"]])
            with col2:
                st.subheader("Mean Wait Time Over Time")
                # Plot the renamed columns
                st.line_chart(chart_data[[f"{selected_model_name} Mean Wait", "Fixed-Time Mean Wait"]])
        
        progress.progress((step + 1) / steps)
        time.sleep(0.1)

    st.success("✅ Simulation finished successfully!")
    st.balloons()
    st.session_state.running = False