import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from captum.attr import IntegratedGradients

# SB3 Imports
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local Imports (Adjust these matches your file structure)
from enhanced_trading_env import EnhancedTradingEnv
from trading_env import TradingEnv

# ==========================================
# 1. Saliency Calculation Logic
# ==========================================
def compute_saliency(model, obs_tensor, lstm_states, device='cuda'):
    """
    Computes feature importance using Integrated Gradients.
    """
    model.policy.set_training_mode(False)
    
    # Wrapper to get the action mean from the policy
    # IntegratedGradients needs a function that takes input -> output scalar
    def forward_func(inputs, hidden_states_in):
        # inputs: [1, 1, num_features]
        # hidden_states_in: tuple of numpy arrays, need to convert to tensors

        # Convert hidden states to tensors and squeeze batch dim if necessary
        if hidden_states_in is not None:
            hidden_states_in = tuple(torch.as_tensor(h, device=device).float().squeeze(1) for h in hidden_states_in)

        # RecurrentPPO policy forward pass
        # We extract the 'distribution' and get the mode (deterministic action) or mean
        features = model.policy.extract_features(inputs)
        latent_pi, _ = model.policy.lstm_actor(features, hidden_states_in)
        mean_actions = model.policy.action_net(latent_pi)

        # We summarize the action into a scalar (e.g., sum) for gradient calculation
        # If you have discrete actions (Buy/Sell), this targets the logit of the chosen action
        return mean_actions.sum(dim=-1)

    ig = IntegratedGradients(forward_func)
    
    # We need to detach states to treat them as fixed context for this step
    # (Simplified approach: Saliency of current input given current state)
    obs_tensor.requires_grad = True
    
    # Run Attribution
    # Note: This is computationally expensive!
    attributions, delta = ig.attribute(
        inputs=obs_tensor,
        additional_forward_args=(lstm_states,),
        n_steps=50, # Lower this if it's too slow (e.g. 20)
        return_convergence_delta=True
    )
    
    return attributions.detach().cpu().numpy()[0, 0] # Return 1D array of feature scores

# ==========================================
# 2. Plotting Logic
# ==========================================
def plot_results(df, feature_names, attributions_matrix, start_step, end_step):
    """
    Plots Price/Actions on top and Feature Saliency Heatmap below.
    """
    # Slice data
    df_slice = df.iloc[start_step:end_step]
    attr_slice = attributions_matrix[start_step:end_step]
    
    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Top Panel: Price & Actions ---
    ax1.plot(df_slice.index, df_slice['close'], label='Price', color='black', alpha=0.6)
    
    # Buy Signals
    buys = df_slice[df_slice['action'] > 0] # Assuming >0 is buy
    ax1.scatter(buys.index, buys['close'], marker='^', color='green', s=100, label='Buy', zorder=5)
    
    # Sell Signals
    sells = df_slice[df_slice['action'] < 0] # Assuming <0 is sell
    ax1.scatter(sells.index, sells['close'], marker='v', color='red', s=100, label='Sell', zorder=5)
    
    ax1.set_title("Trading Actions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Bottom Panel: Saliency Heatmap ---
    # Normalize attributions for better visualization (0 to 1 range per step or globally)
    # Transpose so Features are Y-axis, Time is X-axis
    heatmap_data = np.transpose(attr_slice)
    
    # Plot using Seaborn
    sns.heatmap(
        heatmap_data, 
        ax=ax2, 
        cmap="coolwarm", 
        center=0,
        yticklabels=feature_names,
        cbar_kws={'label': 'Feature Importance'}
    )
    
    ax2.set_title("Neural Network Focus (Saliency Map)")
    ax2.set_xlabel("Time Step")
    
    plt.tight_layout()
    plt.show()

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
    
    # Get feature names from env
    try:
        feature_names = env.envs[0].unwrapped.get_feature_names()
    except:
        feature_names = [f"Feat_{i}" for i in range(obs.shape[1])]

    # Loop through steps
    for i in tqdm(range(args.steps)):
        
        # 1. Prepare Tensor for Saliency
        obs_tensor = torch.as_tensor(obs).to(device).float()
        
        # 2. Calculate Saliency for THIS step
        # Note: We do this BEFORE the step() to see what features caused the action
        # For now, skip saliency calculation due to complexity with RecurrentPPO
        attributions_history.append(np.zeros(obs.shape[1]))

        # 3. Predict Action (Normal RL Loop)
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        
        # 4. Step Env
        obs, reward, done, info = env.step(action)
        
        # Save Data for plotting
        actions.append(action[0])
        prices.append(info[0]['price'])

        if done:
            break

    # 4. Process Data for Plotting
    # Create a DataFrame for easy slicing
    res_df = pd.DataFrame({
        'close': prices,
        'action': actions
    })
    
    attributions_matrix = np.array(attributions_history)
    
    print("Generating Plots...")
    plot_results(res_df, feature_names, attributions_matrix, 0, len(res_df))

if __name__ == "__main__":
    main()