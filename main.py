import argparse
import pandas as pd
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor 

import wandb
from wandb.integration.sb3 import WandbCallback

from trading_env import TradingEnv 

class RealTimeWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RealTimeWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if wandb.run is None: return True

        infos = self.locals["infos"][0]
        
        # 1. Log Scalars every 100 steps
        if self.num_timesteps % 100 == 0:
            current_lr = 0.0
            try: current_lr = self.model.policy.optimizer.param_groups[0]["lr"]
            except: pass

            wandb.log({
                "realtime/portfolio_value": infos.get("portfolio_value", 0),
                "realtime/balance": infos.get("balance", 0),
                "realtime/step_reward": infos.get("reward", 0),
                "realtime/action": infos.get("action", 0),
                "realtime/learning_rate": current_lr,
                "global_step": self.num_timesteps
            })

        # 2. Log Volume Profile Snapshot every 1000 steps
        # We don't want to log this too often as it consumes bandwidth
        if self.num_timesteps % 1000 == 0 and "vp_heatmap" in infos:
            heatmap_data = infos["vp_heatmap"]
            
            # Create a Table for the Line Plot
            # Columns: [Bin Index (0-99), Intensity]
            data = [[x, y] for x, y in enumerate(heatmap_data)]
            table = wandb.Table(data=data, columns=["bin_index", "volume"])
            
            wandb.log({
                "realtime/vp_snapshot": wandb.plot.line(
                    table, "bin_index", "volume", 
                    title=f"Volume Profile Shape (Step {self.num_timesteps})"
                )
            })
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--vp-days", nargs='+', type=int, default=[7, 30]) 
    parser.add_argument("--algo", type=str, default="sac") 
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Load Data
    csv_file = args.data_path if args.data_path else f"{args.pair}_data.csv"
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found. Using Dummy Data.")
        exit(1)
    else:
        print(f"Loading data: {csv_file}")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        if 'date' in df.columns: 
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

    # Initialize Env
    env = DummyVecEnv([lambda df=df: Monitor(TradingEnv(df, vp_days=args.vp_days))])

    # Model
    policy_kwargs = dict(net_arch=[512, 512]) 
    if args.algo.lower() == 'sac':
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"./{args.algo}_tb/", learning_rate=3e-4)
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"./{args.algo}_tb/")

    # Callbacks
    callbacks = []
    if args.wandb:
        wandb.init(project="ai-trading-bot", config=vars(args), sync_tensorboard=True, monitor_gym=True)
        callbacks.append(RealTimeWandbCallback())
        callbacks.append(WandbCallback(verbose=2))

    callbacks.append(CheckpointCallback(save_freq=50000, save_path=f'./models/{args.pair}'))

    print(f"Starting training...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(f"{args.algo}_{args.pair}_final")

if __name__ == "__main__":
    main()