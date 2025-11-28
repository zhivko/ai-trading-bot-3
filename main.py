import argparse
import pandas as pd
import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor # <--- Important wrapper

import wandb
from wandb.integration.sb3 import WandbCallback

# Import your corrected environment
from trading_env import TradingEnv 

# --- CUSTOM CALLBACK TO EXTRACT INFO METRICS ---
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in Tensorboard/WandB.
    It extracts data from the 'info' dictionary returned by the environment.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the 'info' dictionary of the current step
        # self.locals['infos'] is a list (one for each env). We assume 1 env.
        infos = self.locals["infos"][0]
        
        # Explicitly record the values you want to see
        if "portfolio_value" in infos:
            self.logger.record("rollout/portfolio_value", infos["portfolio_value"])
        if "balance" in infos:
            self.logger.record("rollout/balance", infos["balance"])
        if "reward" in infos:
            self.logger.record("rollout/step_reward", infos["reward"])
        if "action" in infos:
            self.logger.record("rollout/action_val", infos["action"])
        if "shares_held" in infos:
            self.logger.record("rollout/shares_held", infos["shares_held"])
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--vp-days", nargs='+', type=int, default=[7, 30])
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # 1. Load Data
    data_path = f"data/{args.pair}.csv" 
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Creating dummy data.")
        # Dummy data for testing
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='H')
        df = pd.DataFrame({'close': [50000 + x + np.sin(x/100)*1000 for x in range(10000)]}, index=dates)
        df['open'] = df['close']
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99
        df['volume'] = 1000 + np.random.rand(10000)*100
    else:
        df = pd.read_csv(data_path)

    # 2. Initialize Environment
    # We must wrap the env in Monitor() to allow SB3 to track stats properly
    env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

    # 3. Initialize Agent (SAC)
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"./{args.algo}_tensorboard/",
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto'
    )

    # 4. Callbacks
    callbacks = []
    
    # Add our Custom Callback to see Net Worth!
    callbacks.append(TensorboardCallback())

    if args.wandb:
        run = wandb.init(
            project="ai-trading-bot", 
            config=vars(args),
            sync_tensorboard=True, # This syncs the self.logger.record calls
            monitor_gym=True,      # This tries to record video/stats if available
            save_code=True,
        )
        callbacks.append(WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f'./models/{args.pair}',
        name_prefix=args.algo
    )
    callbacks.append(checkpoint_callback)

    # 5. Train
    print(f"Starting training for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    
    model.save(f"{args.algo}_{args.pair}_final")
    print("Training complete.")

if __name__ == "__main__":
    main()