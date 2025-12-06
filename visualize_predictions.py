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
from features import get_features
import logging

# 1. Load the trained model
model = RecurrentPPO.load("ppo_crypto_trader.zip")
logging.info("Model loaded successfully")

# 2. Load data
pair = 'BTC/USDT'
data_file = f'{pair.replace("/", "_")}_data.csv'
df = pd.read_csv(data_file, parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
logging.info(f"Full data length: {len(df)} rows")

# Compute VP with caching
vp7_file = f'{pair.replace("/", "_")}_vp7.pkl'
vp30_file = f'{pair.replace("/", "_")}_vp30.pkl'

data_mtime = os.path.getmtime(data_file) if os.path.exists(data_file) else 0
vp7_mtime = os.path.getmtime(vp7_file) if os.path.exists(vp7_file) else 0
vp30_mtime = os.path.getmtime(vp30_file) if os.path.exists(vp30_file) else 0

if os.path.exists(vp7_file) and vp7_mtime >= data_mtime:
    logging.info("Loading cached 7d VP...")
    with open(vp7_file, 'rb') as f:
        vp7_df = pickle.load(f)
else:
    logging.info("Computing 7d VP...")
    vp7_df = get_rolling_vp(df, 7)
    logging.info("Saving 7d VP...")
    with open(vp7_file, 'wb') as f:
        pickle.dump(vp7_df, f)

if os.path.exists(vp30_file) and vp30_mtime >= data_mtime:
    logging.info("Loading cached 30d VP...")
    with open(vp30_file, 'rb') as f:
        vp30_df = pickle.load(f)
else:
    logging.info("Computing 30d VP...")
    vp30_df = get_rolling_vp(df, 30)
    logging.info("Saving 30d VP...")
    with open(vp30_file, 'wb') as f:
        pickle.dump(vp30_df, f)

logging.info("VP computation completed.")

# NO ENV SIMULATION: Direct rollout over FULL df for viz (faster, deterministic)
logging.info("Starting full-data simulation...")
actions = []
values = []
prices = []

# Initialize for recurrent policy
lstm_states = None
episode_starts = torch.tensor([1.0], dtype=torch.float32)

for step in range(len(df)):  # FULL df!
    current_price = df['close'].iloc[step]
    obs = get_features(df, vp7_df, vp30_df, df.index[step])
    
    # Predict action/value
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    lstm_states = (torch.tensor(lstm_states[0], dtype=torch.float32), torch.tensor(lstm_states[1], dtype=torch.float32)) if lstm_states is not None else None
    episode_starts = torch.tensor([0.0], dtype=torch.float32)
    
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        value = model.policy.predict_values(obs_tensor, lstm_states, episode_starts).item()
    
    actions.append(action[0])
    values.append(value)
    prices.append(current_price)
    
    if step % 5000 == 0:
        logging.info(f"Processed {step} steps")

logging.info(f"Full simulation complete. Collected {len(actions)} steps ({len(df)} total)")
# 4. Plot everything beautifully
logging.info("Generating plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)  # Taller for long timeline

vp_window = 24
num_bins = 50
ax1.plot(range(len(prices)), prices, label='Price', color='blue', linewidth=0.8)  # Thinner for long plot

for i in range(vp_window, len(prices)):  # Now full ~40k!
    window_data = df.iloc[i - vp_window : i]
    vp = compute_volume_profile(window_data, num_bins=num_bins)
    
    # POC/VA/HVN (unchanged, but now every step across years)
    ax1.axhline(vp['poc'], xmin=i/len(prices), xmax=(i+1)/len(prices), color='yellow', lw=1, alpha=0.6, label='POC' if i==vp_window else '')
    ax1.fill_between([i, i+1], vp['val'], vp['vah'], color='cyan', alpha=0.1, label='Value Area' if i==vp_window else '')  # Lower alpha for density
    
    # HVN (add plot skip for perf: every 10th to avoid overload)
    if i % 10 == 0:  # ← NEW: Sample for speed (remove for full density)
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
                ax1.scatter(i, level, s=min(vol * 5, 100), marker='^', c='green', alpha=0.5, label='HVN' if i==vp_window else '')  # Smaller s/alpha
# New version – works with continuous actions [-1, 1]
long_entries  = [i for i, a in enumerate(actions) if a > 0.6]      # confident long
short_entries = [i for i, a in enumerate(actions) if a < -0.6]     # confident short

ax1.scatter(long_entries,  [prices[i] for i in long_entries],  marker='^', color='lime',  s=100, edgecolors='black', linewidth=1.5, label='Long Entry', zorder=10, alpha=0.7)
ax1.scatter(short_entries, [prices[i] for i in short_entries], marker='v', color='red',   s=100, edgecolors='black', linewidth=1.5, label='Short Entry', zorder=10, alpha=0.7)

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
            marker='o', facecolors='none', edgecolors='orange', s=80, linewidth=2.5,
            label='Exit Long', zorder=9, alpha=0.7)
ax1.scatter(exit_shorts, [prices[i] for i in exit_shorts],
            marker='o', facecolors='none', edgecolors='purple', s=80, linewidth=2.5,
            label='Exit Short', zorder=9, alpha=0.7)

ax1.set_title("Price + Volume Profile Context (Full 5-Year Rollout)")
ax1.legend()
ax1.grid(alpha=0.2)

ax2.plot(actions, label='Action', color='green', linewidth=0.8)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_title("Agent Actions (-1=Short, 0=Neutral, 1=Long)")
ax2.legend()
ax2.grid(alpha=0.2)

# Fig 2: Now full length
fig2, ax3 = plt.subplots(figsize=(16, 4))
ax3.plot(values, color='orange', linewidth=0.5)
ax3.set_title("Value Function (expected future return)")
ax3.grid(alpha=0.2)

step = max(1, len(df) // 10)  # ~4k for 40k
date_labels = df.index[::step]
for ax in (ax1, ax2, ax3):  # Add to fig2 too
    ax.set_xticks(range(0, len(df), step))
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in date_labels], rotation=45)  # Monthly for long view
ax3.set_xlabel('Steps (Full Timeline)')

plt.tight_layout()
plt.show()