# backtest.py
import torch
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from stable_baselines3 import SAC
import gymnasium as gym
from trading_env import TradingEnv
from volume_profile import get_rolling_vp
import wandb

# 1. Load the trained model
model = SAC.load("sac_crypto_trader.zip")

# 2. Load data
data_file = 'BTC_USDT_data.csv'
df = pd.read_csv(data_file, index_col=0, parse_dates=True)
print("Data loaded successfully. Shape:", df.shape)

# Compute VP with caching
pair = 'BTC/USDT'  # Assuming BTC/USDT for backtest
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

# Initialize wandb for backtest logging
wandb.init(project="grok-crypto-trader", name="backtest-btc-usdt", job_type="backtest")

# 3. Run a deterministic rollout and collect everything
obs, info = env.reset()
done = False

# Initialize LSTM states for recurrent policy
lstm_states = None
episode_starts = torch.tensor([1.0], dtype=torch.float32)  # True as 1.0

actions = []
portfolio_values = []
rewards = []
prices = []
vp_poc = []
vp_vah = []
vp_val = []
step = 0

while not done:
    # Get action + internal tensors
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    # Convert lstm_states to tensor
    lstm_states = (torch.tensor(lstm_states[0], dtype=torch.float32), torch.tensor(lstm_states[1], dtype=torch.float32))

    # Filters to kill bad trades
    lookback = 24
    if len(prices) >= lookback:
        recent_prices = prices[-lookback:]
    else:
        recent_prices = prices
    current_val = vp_val[-1] if vp_val else prices[-1] if prices else env.df.iloc[env.current_step]['close']
    current_vah = vp_vah[-1] if vp_vah else prices[-1] if prices else env.df.iloc[env.current_step]['close']
    hours_below_val = sum(1 for p in recent_prices if p < current_val)
    hours_above_vah = sum(1 for p in recent_prices if p > current_vah)
    entering_long = action[0] > 0 and (len(actions) == 0 or actions[-1] <= 0)
    entering_short = action[0] < 0 and (len(actions) == 0 or actions[-1] >= 0)
    if entering_long and hours_below_val < 8:
        action = np.array([0.0])
    if entering_short and hours_above_vah < 8:
        action = np.array([0.0])

    # Volume profile filter
    t = env.df.index[env.current_step]
    vp7 = vp7_df.loc[t]
    if not pd.isna(vp7['max_volume']) and vp7['max_volume'] > 0:
        bins = vp7['bins']
        vp = vp7['vp']
        min_p = bins[0]
        bin_size = bins[1] - bins[0] if len(bins) > 1 else 0
        close = env.df.iloc[env.current_step]['close']
        if bin_size > 0:
            bin_idx = int((close - min_p) / bin_size)
            if 0 <= bin_idx < len(vp):
                volume_at_price = vp[bin_idx]
                current_node_strength = volume_at_price / vp7['max_volume']
                if current_node_strength < 0.2:
                    action = np.array([0.0])

    # Let the environment itself handle minimum position size
    obs, reward, terminated, truncated, info = env.step(action)   # ← just pass raw action
    done = terminated or truncated

    step += 1
    if step % 1000 == 0:
        print(f"Step {step}: Portfolio value {portfolio_values[-1]}")

    actions.append(action[0])  # action is array
    rewards.append(reward)
    portfolio_values.append(env._calculate_portfolio_value(env.df.iloc[env.current_step]['close']))

    prices.append(env.df.iloc[env.current_step]['close'])
    t = env.df.index[env.current_step]
    vp7 = vp7_df.loc[t]
    vp_poc.append(vp7['poc'] if not pd.isna(vp7['poc']) else env.df.iloc[env.current_step]['close'])
    vp_vah.append(vp7['vah'] if not pd.isna(vp7['vah']) else env.df.iloc[env.current_step]['close'])
    vp_val.append(vp7['val'] if not pd.isna(vp7['val']) else env.df.iloc[env.current_step]['close'])

    episode_starts = torch.tensor([0.0], dtype=torch.float32)  # False as 0.0

# 4. Compute backtest metrics
initial_value = env.initial_cash
final_value = portfolio_values[-1] if portfolio_values else initial_value
total_return = (final_value - initial_value) / initial_value * 100

# Calculate returns
returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else []
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24) if returns.size > 0 else 0  # Assuming hourly data

max_drawdown = 0
peak = initial_value
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak
    if drawdown > max_drawdown:
        max_drawdown = drawdown
max_drawdown *= 100

print(f"Total Return: {total_return:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Final Portfolio Value: ${final_value:.2f}")

# Log backtest metrics to wandb
wandb.log({
    'backtest_total_return_pct': total_return,
    'backtest_sharpe_ratio': sharpe_ratio,
    'backtest_max_drawdown_pct': max_drawdown,
    'backtest_final_portfolio_value': final_value,
    'backtest_num_trades': num_trades,
    'backtest_initial_value': initial_value
})

# Log equity curve as a custom chart
equity_returns = [(pv / initial_value - 1) * 100 for pv in portfolio_values]
wandb.log({"backtest_equity_curve": wandb.plot.line_series(
    xs=list(range(len(equity_returns))),
    ys=[equity_returns],
    keys=["Portfolio Return %"],
    title="Backtest Equity Curve",
    xname="Time Step"
)})

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax1.plot(prices, label='Price', color='blue', linewidth=1.2)
ax1.fill_between(range(len(prices)), vp_val, vp_vah, alpha=0.15, color='cyan', label='Value Area')
ax1.plot(vp_poc, color='yellow', linewidth=1, alpha=0.8, label='POC')
long_entries  = [i for i, a in enumerate(actions) if a > 0.85]      # confident long
short_entries = [i for i, a in enumerate(actions) if a < -0.85]     # confident short

num_trades = len(long_entries) + len(short_entries)
print(f"Number of Trades: {num_trades}")

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

ax1.set_title("Backtest Price with Entries")
ax1.legend()
ax1.grid(alpha=0.3)

# Equity curve
ax2.plot(range(len(portfolio_values)), [(pv / initial_value - 1) * 100 for pv in portfolio_values], label='Portfolio Return %', color='blue')
ax2.set_title("Portfolio Equity Curve")
ax2.legend()
ax2.grid(alpha=0.3)

for ax in (ax1, ax2):
    ax.set_xticks(range(0, len(df), len(df)//20))  # ~20 readable ticks
    ax.set_xticklabels(df.index[::len(df)//20].strftime('%Y-%m-%d %H:%M'),
                       rotation=45, ha='right')

plt.tight_layout()

# Log the backtest plot to wandb
wandb.log({"backtest_plot": wandb.Image(plt)})

plt.show()