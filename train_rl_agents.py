import pandas as pd
import numpy as np
import matplotlib
import talib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from custom_trading_env import ContinuousTradingEnv # Assuming you saved the env above
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0, render_freq=1000):
        super(ImageRecorderCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        # Only run every 1000 steps (or at end of episode) to save speed
        if self.n_calls % self.render_freq == 0:
            # Get the image from the environment
            # Access the unwrapped first env to bypass any wrappers that don't support mode argument
            img = self.training_env.envs[0].unwrapped.render(mode='rgb_array')

            # Log to Weights & Biases
            if img is not None:
                wandb.log({"trading_chart": wandb.Image(img, caption=f"Step {self.num_timesteps}")})
        
        return True


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_net_worths = []
        self.ep_trade_count = 0
        self.ep_buy_count = 0
        self.ep_sell_count = 0
        self.ep_prices = []
        self.ep_actions = []
        self.ep_dates = []
        self.ep_balances = []
        self.ep_shares = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if wandb.run is not None:
            # Log timesteps
            wandb.log({"timesteps": self.num_timesteps})

            # Log SAC entropy coefficient
            if hasattr(self.model, 'ent_coef'):
                wandb.log({"ent_coef": self.model.ent_coef})

            # Extract net_worth from info
            infos = self.locals.get('infos')
            if infos:
                info = infos[0] if isinstance(infos, list) else infos
                net_worth = info.get('net_worth', 0)
                wandb.log({"net_worth": net_worth})

                # Collect for episode metrics
                self.ep_net_worths.append(net_worth)

                # Collect for plotting
                current_price = info.get('current_price', info.get('price', 0))
                action = info.get('action', 0)
                current_date = info.get('timestamp', f"{self.n_calls}")
                self.ep_prices.append(current_price)
                self.ep_actions.append(action)
                self.ep_dates.append(current_date)
                self.ep_balances.append(info.get('balance', 0))
                self.ep_shares.append(info.get('shares_held', 0))

                # Count trades
                if info.get('trade_executed', False):
                    self.ep_trade_count += 1
                    if action > 0:
                        self.ep_buy_count += 1
                    elif action < 0:
                        self.ep_sell_count += 1

            # Generate plot every 100 steps
            if self.num_timesteps % 100 == 0 and len(self.ep_prices) > 10:
                self._generate_and_log_plot()

        return True

    def _on_rollout_end(self):
        if wandb.run is not None:
            self.episode_count += 1
            # Generate episode plot before resetting
            if len(self.ep_balances) > 1:
                self._generate_episode_plot()

            # Log trade counts
            wandb.log({
                "episode_number": self.episode_count,
                "episode_trade_count": self.ep_trade_count,
                "episode_buy_count": self.ep_buy_count,
                "episode_sell_count": self.ep_sell_count
            })

            if len(self.ep_net_worths) > 1:
                # Calculate Sharpe ratio (simplified)
                returns = np.diff(self.ep_net_worths) / self.ep_net_worths[:-1]
                if len(returns) > 0:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(8760)  # Assuming hourly data
                    wandb.log({"sharpe_ratio": sharpe})

            # Reset for next episode
            self.ep_net_worths = []
            self.ep_trade_count = 0
            self.ep_buy_count = 0
            self.ep_sell_count = 0
            self.ep_prices = []
            self.ep_actions = []
            self.ep_dates = []
            self.ep_balances = []
            self.ep_shares = []

    def _generate_and_log_plot(self):
        if len(self.ep_balances) < 2 or wandb.run is None:
            return

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        steps = np.arange(len(self.ep_balances))
        balances = np.array(self.ep_balances)
        shares = np.array(self.ep_shares)
        actions = np.array(self.ep_actions)
        prices = np.array(self.ep_prices)

        # Price plot
        ax1.plot(steps, prices, label='Price', color='black')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Portfolio Visualization at Step {self.num_timesteps}')
        ax1.grid(True)

        # Mark trades
        buy_steps = [i for i, a in enumerate(actions) if a > 0 and i < len(prices)]
        sell_steps = [i for i, a in enumerate(actions) if a < 0 and i < len(prices)]
        ax1.scatter(buy_steps, [prices[i] for i in buy_steps], color='green', marker='^', label='Buy')
        ax1.scatter(sell_steps, [prices[i] for i in sell_steps], color='red', marker='v', label='Sell')
        ax1.legend()

        # Balance plot
        ax2.plot(steps, balances, label='Balance', color='blue')
        ax2.set_ylabel('Balance ($)')
        ax2.grid(True)
        ax2.legend()

        # Shares plot
        ax3.plot(steps, shares, label='Shares Held', color='green')
        ax3.set_ylabel('Shares Held')
        ax3.grid(True)
        ax3.legend()

        # Actions plot
        ax4.bar(steps, actions, color=['green' if a > 0 else 'red' if a < 0 else 'gray' for a in actions])
        ax4.set_ylabel('Action')
        ax4.set_xlabel('Steps')
        ax4.grid(True)

        plt.tight_layout()

        # Log to wandb
        wandb.log({"trading_chart": wandb.Image(fig)})
        print(f"Logged trading chart at step {self.num_timesteps}")

        plt.close(fig)

    def _generate_episode_plot(self):
        if len(self.ep_balances) < 2 or wandb.run is None:
            return

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        steps = np.arange(len(self.ep_balances))
        balances = np.array(self.ep_balances)
        shares = np.array(self.ep_shares)
        actions = np.array(self.ep_actions)
        prices = np.array(self.ep_prices)

        # Price plot
        ax1.plot(steps, prices, label='Price', color='black')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Episode Portfolio Visualization (Ended at Step {self.num_timesteps})')
        ax1.grid(True)

        # Mark trades
        buy_steps = [i for i, a in enumerate(actions) if a > 0 and i < len(prices)]
        sell_steps = [i for i, a in enumerate(actions) if a < 0 and i < len(prices)]
        ax1.scatter(buy_steps, [prices[i] for i in buy_steps], color='green', marker='^', label='Buy')
        ax1.scatter(sell_steps, [prices[i] for i in sell_steps], color='red', marker='v', label='Sell')
        ax1.legend()

        # Balance plot
        ax2.plot(steps, balances, label='Balance', color='blue')
        ax2.set_ylabel('Balance ($)')
        ax2.grid(True)
        ax2.legend()

        # Shares plot
        ax3.plot(steps, shares, label='Shares Held', color='green')
        ax3.set_ylabel('Shares Held')
        ax3.grid(True)
        ax3.legend()

        # Actions plot
        ax4.bar(steps, actions, color=['green' if a > 0 else 'red' if a < 0 else 'gray' for a in actions])
        ax4.set_ylabel('Action')
        ax4.set_xlabel('Steps')
        ax4.grid(True)

        plt.tight_layout()

        # Log to wandb
        wandb.log({"episode_trading_chart": wandb.Image(fig)})
        print(f"Logged episode trading chart at step {self.num_timesteps}")

        plt.close(fig)

# --- CONFIGURATION ---
DATA_FILE = 'BTCUSDT_data.csv'
TIMESTEPS = 500000
WINDOW_SIZE = 35

wandb.init(project="crypto-trading-rl", name="sac-training")

# 1. LOAD AND PREPARE DATA
try:
    df = pd.read_csv(DATA_FILE)
    # Calculate technical indicators using TA-Lib
    df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
    df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
    macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    fastk_3_14, fastd_3_14 = talib.STOCHRSI(df['close'].values, timeperiod=14, fastk_period=3, fastd_period=3)
    df['sto_rsi_3_14'] = fastk_3_14
    fastk_10_60, fastd_10_60 = talib.STOCHRSI(df['close'].values, timeperiod=60, fastk_period=10, fastd_period=3)
    df['sto_rsi_10_60'] = fastk_10_60
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please create it first.")
    exit()

# Simple train/test split
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# 2. CREATE ENVIRONMENT FACTORY FUNCTION
def make_env(data_frame):
    # Pass the specific dataframe slice to the custom environment
    return ContinuousTradingEnv(df=data_frame, window_size=WINDOW_SIZE, commission=0.001, buy_threshold=0.1, sell_threshold=-0.1)

# Create Vectorized Environment for stable training
train_env = make_vec_env(lambda: make_env(train_df), n_envs=4)


# Initialize the callback
# img_callback = ImageRecorderCallback(render_freq=5000) # Render every 2000 steps

# 3. TRAIN SAC (Soft Actor-Critic)
# SAC is off-policy: uses a large replay buffer and is generally more sample efficient.
print("--- Training Soft Actor-Critic (SAC) Agent ---")
sac_model = SAC(
    "MlpPolicy",
    train_env,
    verbose=1,
    buffer_size=1000000,
    learning_rate=3e-5,
    ent_coef='auto' # Automatically manages exploration/exploitation,
)

sac_model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=[WandbCallback()])
sac_model.save("sac_crypto_trader")
print("SAC training complete and model saved.")

'''
# 4. TRAIN PPO (Proximal Policy Optimization)
# PPO is on-policy: simpler, more stable, but requires more steps.
print("\n--- Training Proximal Policy Optimization (PPO) Agent ---")
ppo_model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=1024, # Number of steps before one update
    batch_size=256
)
ppo_model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
ppo_model.save("ppo_crypto_trader")
print("PPO training complete and model saved.")
'''

# In train_rl_agents.py, update the EVALUATE section to match this:

# 5. EVALUATE (Example for SAC)
print("\n--- Evaluating SAC Model on Test Data ---")
eval_env_sac = make_env(test_df) # Non-vectorized for simple evaluation
obs, info = eval_env_sac.reset()
terminated = False
while not terminated:
    # deterministic=True means the agent takes the most confident action
    action, _states = sac_model.predict(obs, deterministic=True) 
    obs, reward, terminated, truncated, info = eval_env_sac.step(action)
    terminated = terminated or truncated

final_net_worth_sac = info['net_worth']
print(f"SAC Final Net Worth on Test Data: ${final_net_worth_sac:,.2f}")
wandb.log({"final_net_worth_sac": final_net_worth_sac})

# NEW: Visualize the SAC results
# The output filename will be sac_evaluation_chart_[timestamp].png
eval_env_sac.render(mode='human', agent_name='SAC') 

'''
# BONUS: Evaluate PPO
print("\n--- Evaluating PPO Model on Test Data ---")
eval_env_ppo = make_env(test_df)
obs, info = eval_env_ppo.reset()
terminated = False
while not terminated:
    action, _states = ppo_model.predict(obs, deterministic=True) 
    obs, reward, terminated, truncated, info = eval_env_ppo.step(action)
    terminated = terminated or truncated

final_net_worth_ppo = info['net_worth']
print(f"PPO Final Net Worth on Test Data: ${final_net_worth_ppo:,.2f}")

# NEW: Visualize the PPO results
# The output filename will be ppo_evaluation_chart_[timestamp].png
eval_env_ppo.render(mode='human', agent_name='PPO')
'''