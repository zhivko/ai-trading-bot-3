# backtest.py — FINAL WORKING VERSION FOR SAC (sac_crypto_trader.zip)
import torch
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
import pickle
import os
from stable_baselines3 import SAC
from trading_env import TradingEnv
from volume_profile import get_rolling_vp

# ========================== 1. LOAD DATA & VP ==========================
pair = 'BTC/USDT'
data_file = f'{pair.replace("/", "_")}_data.csv'
df = pd.read_csv(data_file, parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
print(f"Data loaded: {len(df)} rows")

# VP caching
vp7_file = f'{pair.replace("/", "_")}_vp7.pkl'
vp30_file = f'{pair.replace("/", "_")}_vp30.pkl'

def load_or_compute_vp(days, filename):
    if os.path.exists(filename) and os.path.getmtime(filename) >= os.path.getmtime(data_file):
        print(f"Loading cached {days}d VP...")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Computing {days}d VP...")
        vp = get_rolling_vp(df, days)
        with open(filename, 'wb') as f:
            pickle.dump(vp, f)
        return vp

vp7_df  = load_or_compute_vp(7,  vp7_file)
vp30_df = load_or_compute_vp(30, vp30_file)

# ========================== 2. LOAD SAC MODEL ==========================
print("Loading SAC model...")
model = SAC.load("sac_crypto_trader.zip")
print("Model loaded successfully")

# ========================== 3. ENVIRONMENT ==========================
# Set episode_length_days to cover the full dataset (approx 4 years)
total_days = (len(df) - 30 * 24) // 24  # Available days after VP warmup
env = TradingEnv(df, vp7_df, vp30_df, episode_length_days=total_days)

# ========================== 4. BACKTEST LOOP (SAC) ==========================
obs, _ = env.reset()
done = False

actions = []
prices = []
portfolio_values = []
q_values = []           # Q-value from critics (proxy for value function)

step = 0
print("Starting backtest simulation...")

while not done:
    # Predict action (deterministic)
    action, _ = model.predict(obs, deterministic=True)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # === CORRECT WAY TO GET Q-VALUES FROM SAC ===
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)

        # SAC's critic expects (obs, action) pair
        q_values_batch = model.critic(obs_tensor, action_tensor)  # returns tuple (q1, q2)
        q_value = torch.min(q_values_batch[0], q_values_batch[1]).item()     # conservative Q

    # Record
    current_price = env.df.iloc[env.current_step]['close']
    portfolio_value = env._calculate_portfolio_value(current_price)

    actions.append(float(action[0]))
    q_values.append(q_value)
    prices.append(current_price)
    portfolio_values.append(portfolio_value)

    step += 1
    if step % 1000 == 0:
        print(f"Step {step:,} | Price ${current_price:,.0f} | Portfolio ${portfolio_value:,.0f}")

print(f"Backtest finished! Total steps: {len(actions)}")
start_date = env.df.index[env.start_step]
end_date = env.df.index[env.current_step]
print(f"Dates tested: {start_date} to {end_date}")

# ========================== 5. LOG TO WANDB ==========================
wandb.init(project="grok-crypto-trader", name="backtest-btc-usdt-sac")

total_return_pct = (portfolio_values[-1] / env.initial_cash - 1) * 100
returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24) if len(returns) > 1 else 0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 1])
ax1.plot(portfolio_values, label=f"Equity → ${portfolio_values[-1]:,.0f}", color="purple", linewidth=2)
ax1.set_title(f"SAC Backtest – Total Return: {total_return_pct:+.2f}% | Sharpe: {sharpe:.2f}")
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(actions, color="green", alpha=0.7)
ax2.axhline(0, color="gray", linestyle="--")
ax2.set_ylabel("Action")
ax2.set_xlabel("Step")
ax2.grid(alpha=0.3)

wandb.log({
    "final_portfolio_value": portfolio_values[-1],
    "total_return_%": total_return_pct,
    "sharpe_ratio": sharpe,
    "steps": len(actions),
    "actions_histogram": wandb.Histogram(actions),
    "q_value_mean": np.mean(q_values),
    "equity_curve": wandb.Image(fig)
})

plt.tight_layout()
wandb.log({"full_plot": wandb.Image(fig)})
plt.close(fig)

wandb.finish()
print(f"\nBacktest complete!")
print(f"Final portfolio: ${portfolio_values[-1]:,.0f}")
print(f"Total return: {total_return_pct:+.2f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"W&B run → https://wandb.ai/zhivko/grok-crypto-trader/runs/{wandb.run.id if wandb.run else 'latest'}")