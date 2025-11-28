import argparse
import pandas as pd
import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor 

import wandb
from wandb.integration.sb3 import WandbCallback

# --- FIXED IMPORT HERE ---
from trading_env import TradingEnv 

# --- REAL-TIME WANDB CALLBACK ---
class RealTimeWandbCallback(BaseCallback):
    """
    Logs metrics to WandB every 100 steps, bypassing SB3's episode-end restriction.
    """
    def __init__(self, verbose=0):
        super(RealTimeWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Get the 'info' from the environment
        # self.locals['infos'] is a list (one per env). We assume 1 env.
        infos = self.locals["infos"][0]
        
        # Log every 100 steps (adjust as needed)
        if self.num_timesteps % 100 == 0:
            
            # Extract values safely
            portfolio_value = infos.get("portfolio_value", 0)
            balance = infos.get("balance", 0)
            step_reward = infos.get("reward", 0)
            action_val = infos.get("action", 0)
            shares = infos.get("shares_held", 0)
            price = infos.get("price", 0)

            # Send DIRECTLY to WandB
            wandb.log({
                "realtime/portfolio_value": portfolio_value,
                "realtime/balance": balance,
                "realtime/step_reward": step_reward,
                "realtime/action": action_val,
                "realtime/shares_held": shares,
                "realtime/price": price,
                "global_step": self.num_timesteps
            })
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--wandb", action="store_true") # Ensure this flag is used!
    args = parser.parse_args()

    # 1. Load Data
    data_path = f"data/{args.pair}.csv" 
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Generating dummy data for testing.")
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='H')
        df = pd.DataFrame({'close': [50000 + x + np.sin(x/100)*1000 for x in range(10000)]}, index=dates)
        df['open'] = df['close']
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99
        df['volume'] = 1000 + np.random.rand(10000)*100
    else:
        df = pd.read_csv(data_path)

    # 2. Environment
    # Wrap in Monitor to ensure SB3 tracks internal stats
    env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

    # 3. Model (SAC)
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"./sac_tensorboard/",
        learning_rate=3e-4,
        batch_size=256,
        ent_coef='auto'
    )

    # 4. Callbacks
    callbacks = []
    
    if args.wandb:
        wandb.init(
            project="ai-trading-bot", 
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        # Add the REAL-TIME callback
        callbacks.append(RealTimeWandbCallback())
        # Add standard WandB callback
        callbacks.append(WandbCallback(verbose=2))

    # Save model every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./models/', 
        name_prefix='sac'
    )
    callbacks.append(checkpoint_callback)

    # 5. Train
    print(f"Starting training for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    
    model.save("sac_final")
    print("Training complete.")

if __name__ == "__main__":
    main()