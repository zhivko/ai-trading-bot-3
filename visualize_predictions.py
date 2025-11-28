# visualize_predictions.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from trading_env import TradingEnv
from volume_profile import get_rolling_vp, compute_volume_profile

# 1. Load the trained model
model = RecurrentPPO.load("ppo_crypto_trader.zip")
print("Model loaded successfully")

# 2. Load data
pair = 'BTC/USDT'
data_file = f'{pair.replace("/", "_")}_data.csv'
df = pd.read_csv(data_file, index_col=0, parse_dates=True)
print(f"Data loaded: {len(df)} rows")

# Compute VP with caching
vp7_file = f'{pair.replace("/", "_")}_vp7.pkl'
vp30_file = f'{pair.replace("/", "_")}_vp30.pkl'

data_mtime = os.path.getmtime(data_file) if os.path.exists(data_file) else 0
vp7_mtime = os.path.getmtime(vp7_file) if os.path.exists(vp7_file) else 0
vp30_mtime = os.path.getmtime(vp30_file) if os.path.exists(vp30_file) else 0

if os.path.exists(vp7_file) and vp7_mtime >= data_mtime:
    print("Loading cached 7d VP...")
    with open(vp7_file, 'rb') as f:
        vp7_df = pickle.load(f)
else:
    print("Computing 7d VP...")
    vp7_df = get_rolling_vp(df, 7)
    print("Saving 7d VP...")
    with open(vp7_file, 'wb') as f:
        pickle.dump(vp7_df, f)

if os.path.exists(vp30_file) and vp30_mtime >= data_mtime:
    print("Loading cached 30d VP...")
    with open(vp30_file, 'rb') as f:
        vp30_df = pickle.load(f)
else:
    print("Computing 30d VP...")
    vp30_df = get_rolling_vp(df, 30)
    print("Saving 30d VP...")
    with open(vp30_file, 'wb') as f:
        pickle.dump(vp30_df, f)

print("VP computation completed.")

# Create env
env = TradingEnv(df, vp7_df, vp30_df)

# 3. Run a deterministic rollout and collect everything
obs, info = env.reset()
done = False

# Initialize LSTM states for recurrent policy
lstm_states = None
episode_starts = torch.tensor([1.0], dtype=torch.float32)  # True as 1.0

actions = []
values = []       # critic value
prices = []

step = 0
print("Starting simulation...")
while not done:
    # Get action + internal tensors
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    # Convert lstm_states to tensor
    lstm_states = (torch.tensor(lstm_states[0], dtype=torch.float32), torch.tensor(lstm_states[1], dtype=torch.float32))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Extract value
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        value = model.policy.predict_values(obs_tensor, lstm_states, episode_starts).item()

    actions.append(action[0])  # action is array
    values.append(value)
    prices.append(env.df.iloc[env.current_step]['close'])  # current price

    episode_starts = torch.tensor([0.0], dtype=torch.float32)  # False as 0.0

    step += 1
    if step % 1000 == 0:
        print(f"Processed {step} steps")

print(f"Simulation complete. Collected {len(actions)} steps")
# 4. Plot everything beautifully
print("Generating plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Price + Volume Profile background
vp_window = 24  # Bars for rolling VP
num_bins = 50
ax1.plot(range(len(prices)), prices, label='Price', color='blue', linewidth=1.2)
for i in range(vp_window, len(prices)):
    window_data = df.iloc[i - vp_window : i]
    vp = compute_volume_profile(window_data, num_bins=num_bins)
    # POC
    ax1.axhline(vp['poc'], xmin=i/len(prices), xmax=(i+1)/len(prices), color='yellow', lw=2, alpha=0.8, label='POC' if i==vp_window else '')
    # VA
    ax1.fill_between([i, i+1], vp['val'], vp['vah'], color='cyan', alpha=0.15, label='Value Area' if i==vp_window else '')
    # HVN
    threshold = np.percentile(vp['vp'], 75)
    hvn_indices = np.where(vp['vp'] > threshold)[0]
    if len(hvn_indices) > 0:
        bin_size = (vp['bins'][1] - vp['bins'][0]) if len(vp['bins']) > 1 else 0
        hvn_levels = vp['bins'][hvn_indices] + bin_size / 2
        hvn_volumes = vp['vp'][hvn_indices]
        current_price = prices[i]
        local_range = current_price * 0.15
        for level, vol in zip(hvn_levels, hvn_volumes):
            if abs(level - current_price) > local_range:
                continue
            ax1.scatter(i, level, s=min(vol * 10, 150), marker='^', c='green', alpha=0.7, label='HVN' if i==vp_window and level==hvn_levels[0] else '')
# New version – works with continuous actions [-1, 1]
long_entries  = [i for i, a in enumerate(actions) if a > 0.6]      # confident long
short_entries = [i for i, a in enumerate(actions) if a < -0.6]     # confident short

ax1.scatter(long_entries,  [prices[i] for i in long_entries],  marker='^', color='lime',  s=150, edgecolors='black', linewidth=1.5, label='Long Entry', zorder=10)
ax1.scatter(short_entries, [prices[i] for i in short_entries], marker='v', color='red',   s=150, edgecolors='black', linewidth=1.5, label='Short Entry', zorder=10)

exit_longs  = []
exit_shorts = []

for i in range(1, len(actions)):
    prev = actions[i-1]
    curr = actions[i]
    
    # Long exit: was clearly long and now clearly reducing / flat
    if prev > 0.6 and curr <= 0.4 and (i == 1 or actions[i-2] > 0.6):
        exit_longs.append(i)
    
    # Short exit: was clearly short and now reducing / flat
    if prev < -0.6 and curr >= -0.4 and (i == 1 or actions[i-2] < -0.6):
        exit_shorts.append(i)

# Plot them — now they will actually appear!
ax1.scatter(exit_longs,  [prices[i] for i in exit_longs],
            marker='o', facecolors='none', edgecolors='orange', s=120, linewidth=2.5,
            label='Exit Long', zorder=9)
ax1.scatter(exit_shorts, [prices[i] for i in exit_shorts],
            marker='o', facecolors='none', edgecolors='purple', s=120, linewidth=2.5,
            label='Exit Short', zorder=9)

ax1.set_title("Price + Volume Profile Context")
ax1.legend()
ax1.grid(alpha=0.3)

# Agent actions (continuous: -1 short to 1 long)
ax2.plot(actions, label='Action', color='green')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_title("Agent Actions (-1=Short, 0=Neutral, 1=Long)")
ax2.legend()
ax2.grid(alpha=0.3)

# Critic value function in a separate figure
fig2, ax3 = plt.subplots(figsize=(16, 4))
ax3.plot(values, color='orange')
ax3.set_title("Value Function (expected future return)")
ax3.grid(alpha=0.3)

for ax in (ax1, ax2):
    ax.set_xticks(range(0, len(df), len(df)//20))  # ~20 readable ticks
    ax.set_xticklabels(df.index[::len(df)//20].strftime('%Y-%m-%d %H:%M'),
                        rotation=45, ha='right')

plt.tight_layout()
plt.show()