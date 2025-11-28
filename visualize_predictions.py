# visualize_predictions.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from trading_env import TradingEnv
from volume_profile import get_rolling_vp

# 1. Load the trained model
model = RecurrentPPO.load("ppo_crypto_trader.zip")
print("Model loaded successfully")

# 2. Load data
pair = 'BTC/USDT'
data_file = f'{pair.replace("/", "_")}_data.csv'
df = pd.read_csv(data_file, index_col=0, parse_dates=True)
print(f"Data loaded: {len(df)} rows")

# Compute VP
vp7_df = get_rolling_vp(df, 7)
vp30_df = get_rolling_vp(df, 30)
print("Volume profiles computed")

# Create env
env = TradingEnv(df, vp7_df, vp30_df)

# 3. Run a deterministic rollout and collect everything
obs = env.reset()
done = False

# Initialize LSTM states for recurrent policy
lstm_states = None
episode_starts = torch.tensor([1.0], dtype=torch.float32)  # True as 1.0

actions = []
values = []       # critic value
prices = []
vp_poc = []       # Point of Control from volume profile
vp_vah = []
vp_val = []

step = 0
print("Starting simulation...")
while not done:
    # Get action + internal tensors
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    # Convert lstm_states to tensor
    lstm_states = (torch.tensor(lstm_states[0], dtype=torch.float32), torch.tensor(lstm_states[1], dtype=torch.float32))
    obs, reward, done, info = env.step(action)

    # Extract value
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        value = model.policy.predict_values(obs_tensor, lstm_states, episode_starts).item()

    actions.append(action[0])  # action is array
    values.append(value)
    prices.append(env.df.iloc[env.current_step]['close'])  # current price
    # For VP, need to get from vp_df
    t = env.df.index[env.current_step]
    vp7 = vp7_df.loc[t]
    vp30 = vp30_df.loc[t]
    vp_poc.append(vp7['poc'] if not pd.isna(vp7['poc']) else env.df.iloc[env.current_step]['close'])
    vp_vah.append(vp7['vah'] if not pd.isna(vp7['vah']) else env.df.iloc[env.current_step]['close'])
    vp_val.append(vp7['val'] if not pd.isna(vp7['val']) else env.df.iloc[env.current_step]['close'])

    episode_starts = torch.tensor([0.0], dtype=torch.float32)  # False as 0.0

    step += 1
    if step % 1000 == 0:
        print(f"Processed {step} steps")

print(f"Simulation complete. Collected {len(actions)} steps")
# 4. Plot everything beautifully
print("Generating plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Price + Volume Profile background
ax1.plot(prices, label='Price', color='blue', linewidth=1.2)
ax1.fill_between(range(len(prices)), vp_val, vp_vah, alpha=0.15, color='cyan', label='Value Area')
ax1.plot(vp_poc, color='yellow', linewidth=1, alpha=0.8, label='POC')
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