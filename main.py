import argparse
import pandas as pd
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# Import your corrected environment
from trading_env import TradingEnv 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--vp-days", nargs='+', type=int, default=[7, 30])
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # 1. Load Data
    # Replace this with your actual data loading logic
    data_path = f"data/{args.pair}.csv" 
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Creating dummy data for test.")
        # Create dummy data if file missing (for testing the code)
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='H')
        df = pd.DataFrame({'close': [50000 + x + (x%50)*100 for x in range(10000)]}, index=dates)
        # Add other columns expected by your env
        df['open'] = df['close']
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99
        df['volume'] = 1000
    else:
        df = pd.read_csv(data_path)

    # 2. Initialize Environment
    # We wrap it in DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([lambda: TradingEnv(df)])

    # 3. Initialize Agent (SAC)
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"./{args.algo}_tensorboard/",
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto' # Important for SAC to manage exploration
    )

    # 4. Callbacks (WandB + Checkpoints)
    callbacks = []
    
    if args.wandb:
        run = wandb.init(
            project="ai-trading-bot", 
            config=vars(args),
            sync_tensorboard=True
        )
        callbacks.append(WandbCallback())

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f'./models/{args.pair}',
        name_prefix=args.algo
    )
    callbacks.append(checkpoint_callback)

    # 5. Train
    print(f"Starting training for {args.total_timesteps} steps...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving model...")
    
    model.save(f"{args.algo}_{args.pair}_final")
    print("Training complete.")

if __name__ == "__main__":
    main()