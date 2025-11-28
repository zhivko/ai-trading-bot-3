import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
import matplotlib.pyplot as plt
import pickle
import os
import torch
import argparse
from data_fetcher import fetch_data_for_pairs, PAIRS
from volume_profile import get_rolling_vp
from trading_env import TradingEnv
import gymnasium as gym

# to train shorter
# python main.py --pair BTC-USDT --vp-days 7 30 --algo sac --population 12 --wandb
# to train longer
# python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --total-timesteps 5000000 --wandb

# to visualize
# python visualize_predictions.py

# to backtest
# python backtest.py

# to visualize ML process
# https://wandb.ai/zhivko/grok-crypto-trader

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, vp7_df=None, vp30_df=None, log_interval=1000):
        super(WandbCallback, self).__init__(verbose)
        self.portfolio_values = []
        self.rewards = []
        self.vp7_df = vp7_df
        self.vp30_df = vp30_df
        self.log_interval = log_interval

    def _on_step(self):
        # Log per-step reward
        reward = self.locals['rewards'][0] if self.locals['rewards'] else 0
        self.rewards.append(reward)
        wandb.log({'step_reward': reward}, step=self.num_timesteps)

        # Log portfolio value (equity curve)
        if hasattr(self.locals['env'], 'envs'):
            env = self.locals['env'].envs[0]
        else:
            env = self.locals['env']
        if hasattr(env, '_calculate_portfolio_value'):
            current_price = env.df.iloc[env.current_step]['close']
            portfolio_value = env._calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            wandb.log({'portfolio_value': portfolio_value}, step=self.num_timesteps)

        # Log VP heatmap periodically
        if self.num_timesteps % self.log_interval == 0 and self.vp7_df is not None:
            self._log_vp_heatmap(env)

        # Log episode-level metrics when episode ends
        for info in self.locals['infos']:
            if 'episode' in info:
                wandb.log({
                    'episode_reward': info['episode']['r'],
                    'episode_length': info['episode']['l']
                }, step=self.num_timesteps)
        return True

    def _log_vp_heatmap(self, env):
        """Log volume profile heatmap as image"""
        try:
            t = env.df.index[env.current_step]
            vp7 = self.vp7_df.loc[t]
            if not pd.isna(vp7['max_volume']) and vp7['max_volume'] > 0:
                bins = vp7['bins']
                vp = vp7['vp']

                # Use bin centers for plotting (bins are edges, vp are values between edges)
                bin_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2 if len(bins) > 1 else bins
                bar_width = bins[1] - bins[0] if len(bins) > 1 else 1

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(bin_centers, vp, width=bar_width, alpha=0.7, color='blue')
                ax.axhline(y=vp7['max_volume'], color='red', linestyle='--', alpha=0.7, label='Max Volume')
                if not pd.isna(vp7['poc']):
                    ax.axvline(x=vp7['poc'], color='yellow', linewidth=2, label='POC')
                if not pd.isna(vp7['vah']):
                    ax.axvline(x=vp7['vah'], color='green', linewidth=2, label='VAH')
                if not pd.isna(vp7['val']):
                    ax.axvline(x=vp7['val'], color='orange', linewidth=2, label='VAL')
                ax.set_title(f'Volume Profile Heatmap (7-day) at Step {self.num_timesteps}')
                ax.set_xlabel('Price')
                ax.set_ylabel('Volume')
                ax.legend()
                ax.grid(alpha=0.3)

                wandb.log({"vp_heatmap_7d": wandb.Image(fig)}, step=self.num_timesteps)
                plt.close(fig)

            # Log 30-day VP if available
            if self.vp30_df is not None:
                vp30 = self.vp30_df.loc[t]
                if not pd.isna(vp30['max_volume']) and vp30['max_volume'] > 0:
                    bins = vp30['bins']
                    vp = vp30['vp']

                    # Use bin centers for plotting (bins are edges, vp are values between edges)
                    bin_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2 if len(bins) > 1 else bins
                    bar_width = bins[1] - bins[0] if len(bins) > 1 else 1

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(bin_centers, vp, width=bar_width, alpha=0.7, color='purple')
                    ax.axhline(y=vp30['max_volume'], color='red', linestyle='--', alpha=0.7, label='Max Volume')
                    if not pd.isna(vp30['poc']):
                        ax.axvline(x=vp30['poc'], color='yellow', linewidth=2, label='POC')
                    if not pd.isna(vp30['vah']):
                        ax.axvline(x=vp30['vah'], color='green', linewidth=2, label='VAH')
                    if not pd.isna(vp30['val']):
                        ax.axvline(x=vp30['val'], color='orange', linewidth=2, label='VAL')
                    ax.set_title(f'Volume Profile Heatmap (30-day) at Step {self.num_timesteps}')
                    ax.set_xlabel('Price')
                    ax.set_ylabel('Volume')
                    ax.legend()
                    ax.grid(alpha=0.3)

                    wandb.log({"vp_heatmap_30d": wandb.Image(fig)}, step=self.num_timesteps)
                    plt.close(fig)
        except Exception as e:
            print(f"Error logging VP heatmap: {e}")

def main():
    import os
    parser = argparse.ArgumentParser(description='Train crypto trading agent')
    parser.add_argument('--pair', type=str, default='BTC/USDT', help='Trading pair (e.g., BTC/USDT)')
    parser.add_argument('--vp-days', type=int, nargs=2, default=[7, 30], help='Volume profile days (e.g., 7 30)')
    parser.add_argument('--algo', type=str, default='sac', help='Algorithm (sac or ppo)')
    parser.add_argument('--total-timesteps', type=int, default=50000, help='Total timesteps for training')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()

    pair = args.pair
    data_file = f'{pair.replace("/", "_")}_data.csv'
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print("Loaded data from file.")
    else:
        # Fetch data
        start_date = datetime.utcnow() - timedelta(days=120)  # 120 days for 30d VP
        data = fetch_data_for_pairs([pair], start_date)
        df = data[pair]
        df.to_csv(data_file)
        print("Fetched and saved data.")

    # Compute VP with caching
    vp_days = args.vp_days
    vp7_file = f'{pair.replace("/", "_")}_vp{vp_days[0]}.pkl'
    vp30_file = f'{pair.replace("/", "_")}_vp{vp_days[1]}.pkl'

    # Check if VP files exist and are newer than data file
    data_mtime = os.path.getmtime(data_file) if os.path.exists(data_file) else 0
    vp7_mtime = os.path.getmtime(vp7_file) if os.path.exists(vp7_file) else 0
    vp30_mtime = os.path.getmtime(vp30_file) if os.path.exists(vp30_file) else 0

    if os.path.exists(vp7_file) and vp7_mtime >= data_mtime:
        print(f"Loading cached {vp_days[0]}d VP...")
        with open(vp7_file, 'rb') as f:
            vp7_df = pickle.load(f)
    else:
        print(f"Computing {vp_days[0]}d VP...")
        vp7_df = get_rolling_vp(df, vp_days[0])
        print(f"Saving {vp_days[0]}d VP...")
        with open(vp7_file, 'wb') as f:
            pickle.dump(vp7_df, f)

    if os.path.exists(vp30_file) and vp30_mtime >= data_mtime:
        print(f"Loading cached {vp_days[1]}d VP...")
        with open(vp30_file, 'rb') as f:
            vp30_df = pickle.load(f)
    else:
        print(f"Computing {vp_days[1]}d VP...")
        vp30_df = get_rolling_vp(df, vp_days[1])
        print(f"Saving {vp_days[1]}d VP...")
        with open(vp30_file, 'wb') as f:
            pickle.dump(vp30_df, f)

    # === ENVIRONMENT WITH SINGLE LONG EPISODE (THE WINNING SETUP) ===
    env = DummyVecEnv([lambda: TradingEnv(df, vp7_df, vp30_df)])
    
    # Force no random resets + single episode
    env = VecNormalize(
        env,
        norm_obs=True,       # ← keep observations normalized
        norm_reward=False,   # ← reward normalization kills trading bots
        clip_obs=10.0,
    )
    
    # This line is the magic bullet
    env.env_method("reset")  # ensures it starts from step 500 every time SB3 thinks it’s a new episode

    # Initialize wandb if enabled
    use_wandb = args.wandb
    if use_wandb:
        wandb.init(project="grok-crypto-trader", name=f"{args.algo}-baseline")

    if args.algo != 'sac':
        raise ValueError("Only 'sac' algorithm is supported currently")

    # Define hyperparameters for SAC
    hyperparams = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': (1, 'episode'),
        'gradient_steps': 1,
        'total_timesteps': args.total_timesteps,
        'pair': pair,
        'vp_days': args.vp_days,
        'algo': args.algo
    }

    # Log hyperparameters
    if use_wandb:
        wandb.config.update(hyperparams)

    # Set device for GPU support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        try:
            # Test if CUDA actually works by creating a small tensor
            torch.cuda.init()
            test_tensor = torch.randn(1).cuda()
            print(f"Using device: {device}")
        except Exception as e:
            print(f"CUDA detected but not compatible: {e}")
            device = 'cpu'
            print(f"Falling back to device: {device}")
    else:
        print(f"Using device: {device}")

    # Create PPO agent with LSTM policy for memory
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch.nn as nn

    class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=256):
            super().__init__(observation_space, features_dim)
            self.lstm = nn.LSTM(observation_space.shape[0], 128, batch_first=True)
            self.linear = nn.Linear(128, features_dim)

        def forward(self, observations):
            # observations shape: (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            if observations.dim() == 2:
                # Single timestep, add sequence dimension
                observations = observations.unsqueeze(1)
            lstm_out, _ = self.lstm(observations)
            return self.linear(lstm_out[:, -1, :])  # Take last timestep output

    policy_kwargs = dict(
        features_extractor_class=CustomLSTMFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./sac_tensorboard/",
            policy_kwargs=policy_kwargs,
            device=device,
            **{k: v for k, v in hyperparams.items() if k in ['learning_rate', 'buffer_size', 'batch_size', 'tau', 'gamma', 'train_freq', 'gradient_steps']}
        )
    except RuntimeError as e:
        if 'CUDA' in str(e) and device == 'cuda':
            print(f"CUDA error during model initialization: {e}")
            print("Falling back to CPU...")
            device = 'cpu'
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./sac_tensorboard/",
                policy_kwargs=policy_kwargs,
                device=device,
                **{k: v for k, v in hyperparams.items() if k in ['learning_rate', 'buffer_size', 'batch_size', 'tau', 'gamma', 'train_freq', 'gradient_steps']}
            )
        else:
            raise

    # Train
    print(f"Starting {args.algo.upper()} training...")
    callback = WandbCallback(vp7_df=vp7_df, vp30_df=vp30_df) if use_wandb else None
    model.learn(total_timesteps=hyperparams['total_timesteps'], log_interval=10, callback=callback)

    # Save model and VecNormalize stats
    model.save("sac_crypto_trader")
    env.save("vec_normalize.pkl")

    print("SAC Training completed.")

if __name__ == "__main__":
    main()