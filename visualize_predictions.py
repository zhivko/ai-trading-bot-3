import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
from captum.attr import IntegratedGradients

# SB3 Imports
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local Imports (Adjust these matches your file structure)
from enhanced_trading_env import EnhancedTradingEnv
from trading_env import TradingEnv
from fetch_metrics import generate_metrics

# ==========================================
# 1. Saliency Calculation Logic
# ==========================================
def compute_saliency(model, obs_tensor, lstm_states, device='cuda'):
    """
    Computes feature importance using Integrated Gradients.
    """
    was_training = model.policy.training
    model.policy.train()

    # Wrapper to get the action mean from the policy
    # IntegratedGradients needs a function that takes input -> output scalar
    def forward_func(inputs):
        batch_size = inputs.shape[0]

        if lstm_states is not None:
            h_np, c_np = lstm_states
            h_expanded = torch.as_tensor(h_np, device=device).repeat(1, batch_size, 1)
            c_expanded = torch.as_tensor(c_np, device=device).repeat(1, batch_size, 1)
        else:
            # Get dimensions
            num_layers = 2  # Assume 2 layers as per error
            hidden_size = model.policy.lstm_actor.hidden_size
            h_expanded = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            c_expanded = torch.zeros(num_layers, batch_size, hidden_size, device=device)

        starts_expanded = torch.zeros(batch_size, dtype=torch.float, device=device)  # not episode start

        results = model.policy.get_distribution(inputs, (h_expanded, c_expanded), starts_expanded)

        if isinstance(results, tuple):
            distribution = results[0]
        else:
            distribution = results

        action_mean = distribution.mode()
        return action_mean.sum(dim=1)

    ig = IntegratedGradients(forward_func)

    # We need to detach states to treat them as fixed context for this step
    # (Simplified approach: Saliency of current input given current state)
    obs_tensor.requires_grad = True

    # Run Attribution
    # Note: This is computationally expensive!
    attributions, delta = ig.attribute(
        inputs=obs_tensor,
        n_steps=50, # Lower this if it's too slow (e.g. 20)
        return_convergence_delta=True
    )

    model.policy.train(was_training)
    return attributions.detach().cpu().numpy()[0].squeeze() # Return 1D array of feature scores

# ==========================================
# 2. Plotting Logic
# ==========================================

def plot_results(df, feature_names, attributions_matrix, start_step, end_step, first_timestamp, last_timestamp):
    """
    Plots Price/Actions, NN Actions, and Feature Saliency Heatmap in main visualization, and Average Saliency as separate plot.
    """
    # Slice data
    df_slice = df.iloc[start_step:end_step]
    attr_slice = attributions_matrix[start_step:end_step]

    final_net_worth = df_slice['net_worth'].iloc[-1]

    # Setup Main Plot with 3 subplots (no shared x-axis so each shows labels)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [2, 1, 2]})

    # --- Top Panel: Price & Actions ---
    ax1.plot(df_slice.index, df_slice['close'], label='Price', color='black', alpha=0.6)

    # Buy Signals
    buys = df_slice[df_slice['action'] > 0] # Assuming >0 is buy
    ax1.scatter(buys.index, buys['close'], marker='^', color='green', s=100, label='Buy', zorder=5)

    # Sell Signals
    sells = df_slice[df_slice['action'] < 0] # Assuming <0 is sell
    ax1.scatter(sells.index, sells['close'], marker='v', color='red', s=100, label='Sell', zorder=5)

    ax1.set_title(f"Trading Actions from {first_timestamp.strftime('%d.%m.%Y %H:%M:%S')} to {last_timestamp.strftime('%d.%m.%Y %H:%M:%S')} - Final Net Worth: ${final_net_worth:.2f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis labels based on first and last dates
    ax1.set_xticks([df_slice.index[0], df_slice.index[-1]])
    ax1.set_xticklabels([first_timestamp.strftime('%d.%m.%Y %H:%M:%S'), last_timestamp.strftime('%d.%m.%Y %H:%M:%S')])
    ax1.tick_params(axis='x', rotation=45, labelsize=9)

    # --- Second Panel: NN Actions ---
    ax2.plot(df_slice.index, df_slice['action'], label='NN Action', color='blue')
    ax2.set_title("Neural Network Actions")
    ax2.set_ylabel("Action Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set x-axis labels based on first and last dates
    ax2.set_xticks([df_slice.index[0], df_slice.index[-1]])
    ax2.set_xticklabels([first_timestamp.strftime('%d.%m.%Y %H:%M:%S'), last_timestamp.strftime('%d.%m.%Y %H:%M:%S')])
    ax2.tick_params(axis='x', rotation=0, labelsize=9)

    # --- Third Panel: Saliency Heatmap ---
    # Normalize attributions for better visualization (0 to 1 range per step or globally)
    # Transpose so Features are Y-axis, Time is X-axis
    heatmap_data = attr_slice.T  # Features x Time (shape: features, time)

    # === FIXED: Numerical edges via date2num + robust labeling ===
    x_centers_num = mdates.date2num(df_slice.index)  # Convert centers to floats
    
    # Create edges: [first, centers[1:], last + step_interval]
    if len(x_centers_num) > 1:
        step_interval = np.mean(np.diff(x_centers_num))  # Average step in days
        x_edges_num = np.append(x_centers_num[0], x_centers_num[1:] + step_interval / 2)
        x_edges_num = np.append(x_edges_num, x_edges_num[-1] + step_interval / 2)
    else:
        x_edges_num = [x_centers_num[0] - 0.5, x_centers_num[0] + 0.5]  # Single point fallback
    
    # Now safe: All numerical
    ax3.pcolormesh(x_edges_num, np.arange(len(feature_names) + 1), heatmap_data,
                   cmap="coolwarm", norm=mcolors.CenteredNorm())

    ax3.set_yticks(np.arange(len(feature_names)))
    ax3.set_yticklabels(feature_names)

    # Set x-axis labels based on first and last dates
    ax3.set_xticks([mdates.date2num(first_timestamp), mdates.date2num(last_timestamp)])
    ax3.set_xticklabels([first_timestamp.strftime('%d.%m.%Y %H:%M:%S'), last_timestamp.strftime('%d.%m.%Y %H:%M:%S')])

    ax3.tick_params(axis='x', rotation=0, labelsize=9)

    ax3.set_title("Neural Network Focus (Saliency Heatmap)")
    ax3.set_xlabel("Date")
    ax3.tick_params(axis='y', labelsize=4)  # Smaller font for y-axis
    ax3.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/visualization.png', dpi=600, bbox_inches='tight')
    print("Main plot saved to results/visualization.png")

    # --- Separate Average Saliency Plot ---
    fig2, ax4 = plt.subplots(figsize=(10, 8))
    avg_attr = np.mean(np.abs(attributions_matrix), axis=0)
    df_avg = pd.DataFrame({'Feature': feature_names[:len(avg_attr)], 'Avg_Importance': avg_attr})
    df_avg = df_avg.sort_values('Avg_Importance', ascending=False).head(20)

    sns.barplot(x='Avg_Importance', y='Feature', data=df_avg, palette='viridis', ax=ax4)
    ax4.set_title("Average Feature Saliency")
    ax4.set_xlabel("Average Absolute Importance")
    ax4.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.savefig('results/average_saliency.png', dpi=600, bbox_inches='tight')
    print("Average saliency plot saved to results/average_saliency.png")

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--model-path", type=str, default="models/recurrentppo_BTCUSDT.zip", help="Path to .zip model file")
    parser.add_argument("--data-path", type=str, default="BTCUSDT_data.csv")
    parser.add_argument("--steps", type=int, default=200, help="How many steps to visualize")
    parser.add_argument("--start-index", type=int, default=1000, help="Where to start in the dataset")
    args = parser.parse_args()

    # 1. Setup Environment
    # NOTE: You must recreate the env exactly as it was during training
    # This might require adjusting arguments to match your main.py
    print("Setting up environment...")
    
    # Load Dataframe (Adapt to your data loading logic)
    df = pd.read_csv(args.data_path)
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Initialize Env
    # For RecurrentPPO, use lookback_window=1 since LSTM handles sequences
    lookback_window = 1 if "recurrentppo" in args.model_path.lower() else 50
    env = EnhancedTradingEnv(
        df=df,
        lookback_window=lookback_window, # Must match training
    )
    env = DummyVecEnv([lambda: env])
    # Load VecNormalize stats if available (saved alongside the model)
    import os
    vec_normalize_path = args.model_path.replace('.zip', '.pkl')
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env) # Important: Load stats if you used VecNormalize!
        env.training = False # Don't update stats during test
    else:
        print(f"Warning: VecNormalize stats not found at {vec_normalize_path}. Proceeding without normalization.")
    
    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecurrentPPO.load(args.model_path, env=env, device=device)

    # 3. Run Simulation & Collect Saliency
    print("Running simulation and calculating saliency (this may take a moment)...")
    
    obs = env.reset()
    
    # LSTM States (num_layers, batch_size, hidden_size)
    # Initialize with zeros
    lstm_states = None
    
    # Trackers
    actions = []
    attributions_history = []
    prices = []
    net_worths = []
    timestamps = []
    first_timestamp = None
    last_timestamp = None
    
    # Get feature names from env
    try:
        feature_names = env.envs[0].unwrapped.get_feature_names()
    except:
        feature_names = [f"Feat_{i}" for i in range(obs.shape[1])]

    # Loop through steps
    for i in tqdm(range(args.steps)):

        # 1. Prepare Tensor for Saliency
        obs_tensor = torch.as_tensor(obs).to(device).float().unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)

        # 2. Calculate Saliency for THIS step
        # Note: We do this BEFORE the step() to see what features caused the action
        if lstm_states is not None:
            lstm_states_np = (np.asarray(lstm_states[0]), np.asarray(lstm_states[1]))
        else:
            lstm_states_np = None
        attributions = compute_saliency(model, obs_tensor, lstm_states_np, device)
        attributions_history.append(attributions)

        # 3. Predict Action (Normal RL Loop)
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)

        # 4. Get timestamp before step
        timestamps.append(env.envs[0].unwrapped.raw_df['timestamp'].iloc[env.envs[0].unwrapped.current_step])

        # Output step 1 datetime to console
        if i == 0:
            first_timestamp = timestamps[-1]
            print(f"Step 1 datetime: {first_timestamp}")

        # 5. Step Env
        obs, reward, done, info = env.step(action)

        # Save Data for plotting
        actions.append(action[0])
        prices.append(info[0]['price'])
        net_worths.append(info[0]['net_worth'])

        if done:
            break

    last_timestamp = timestamps[-1] if timestamps else None

    # 4. Process Data for Plotting
    # Create a DataFrame for easy slicing
    res_df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'action': actions,
        'net_worth': net_worths
    })
    res_df.set_index('timestamp', inplace=True)
    res_df.index = pd.to_datetime(res_df.index)
    
    attributions_matrix = np.array(attributions_history)
    
    print("Generating Plots...")
    plot_results(res_df, feature_names, attributions_matrix, 0, len(res_df), first_timestamp, last_timestamp)

    print("Generating metrics report...")
    generate_metrics()

if __name__ == "__main__":
    main()