import argparse
import pandas as pd
import os
import numpy as np
import warnings
import torch

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor 

import wandb
from wandb.integration.sb3 import WandbCallback

from trading_env import TradingEnv 

# --- REAL-TIME CALLBACK (For Training Env) ---
class RealTimeWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RealTimeWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if wandb.run is None: return True
        
        # Only log from the TRAINING environment
        infos = self.locals["infos"][0]
        
        if self.num_timesteps % 100 == 0:
            current_lr = 0.0
            try: current_lr = self.model.policy.optimizer.param_groups[0]["lr"]
            except: pass

            current_alpha = 0.0
            if hasattr(self.model, "log_ent_coef"):
                try: current_alpha = np.exp(self.model.log_ent_coef.detach().cpu().item())
                except: pass

            # Attempt to get Q-Values for debug
            q_val = 0.0
            try:
                # Accessing internal replay buffer data is tricky in callbacks
                # We skip complex extraction to avoid crashes during eval
                pass
            except: pass

            wandb.log({
                "realtime/portfolio_value": infos.get("portfolio_value", 0),
                "realtime/balance": infos.get("balance", 0),
                "realtime/step_reward": infos.get("reward", 0),
                "realtime/action": infos.get("action", 0),
                "realtime/alpha_entropy": current_alpha,
                "realtime/learning_rate": current_lr,
                "global_step": self.num_timesteps
            })
            
        # Snapshot Heatmap every 2000 steps
        if self.num_timesteps % 2000 == 0 and "vp_heatmap" in infos:
            heatmap = infos["vp_heatmap"]
            price_bins = infos.get("vp_bins", []) 
            current_price = infos.get("price", 0)

            if len(price_bins) == len(heatmap):
                data = []
                for p, vol in zip(price_bins, heatmap):
                    # Highlight current price with a marker
                    is_current = 0.04 if abs(p - current_price) < (price_bins[1]-price_bins[0]) else 0
                    data.append([p, vol, is_current])
                
                table = wandb.Table(data=data, columns=["price", "volume", "curr_marker"])
                wandb.log({
                    "realtime/vp_snapshot": wandb.plot.line(
                        table, "price", "volume", 
                        title=f"Volume Profile @ ${current_price:.0f}"
                    )
                })
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--data-path", type=str, default="BTCUSDT_data.csv", help="Path to CSV")
    parser.add_argument("--vp-days", nargs='+', type=int, default=[7, 30]) 
    parser.add_argument("--algo", type=str, default="sac") 
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    # NEW: Date to split data
    parser.add_argument("--test-split", type=str, default="2024-01-01", help="Date to split Train/Test")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # --- 1. LOAD & SPLIT DATA ---
    # Fallback to default name if None
    csv_file = args.data_path if args.data_path else "BTCUSDT_data.csv"
    
    print(f"\n--- 1. LOADING DATA: {csv_file} ---")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        if 'timestamp' in df.columns: df.rename(columns={'timestamp': 'date'}, inplace=True)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # --- SPLITTING ---
        print(f"✂️ Splitting data at {args.test_split}...")
        train_df = df[df.index < args.test_split].copy()
        test_df = df[df.index >= args.test_split].copy()
        
        print(f"   📘 Training Data: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"   qh Testing Data:  {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")
        
        if len(test_df) < 1000:
            print("⚠️ WARNING: Test set is very small. Evaluation might be unreliable.")
            
    else:
        print("❌ Error: File not found. Cannot proceed with Split Training.")
        return

    # --- 2. INITIALIZE ENVIRONMENTS ---
    print("\n--- 2. INITIALIZING ENVIRONMENTS ---")
    
    # Training Environment (2021-2023)
    # Note: We create a separate Volume Profile cache for this subset automatically
    print("-> Setting up Training Env...")
    env = DummyVecEnv([lambda train_df=train_df: Monitor(TradingEnv(train_df, vp_days=args.vp_days))])
    
    # Evaluation Environment (2024)
    print("-> Setting up Evaluation Env...")
    eval_env = DummyVecEnv([lambda test_df=test_df: Monitor(TradingEnv(test_df, vp_days=args.vp_days))])

    # --- 3. MODEL SETUP ---
    policy_kwargs = dict(net_arch=[512, 512]) 
    
    if args.algo.lower() == 'sac':
        model = SAC(
            "MlpPolicy", 
            env, 
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=f"./{args.algo}_tb/", 
            learning_rate=3e-4,
            ent_coef='auto'
        )
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"./{args.algo}_tb/")

    # --- 4. CALLBACKS ---
    callbacks = []
    
    if args.wandb:
        wandb.init(
            project="ai-trading-bot", 
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(RealTimeWandbCallback())
        callbacks.append(WandbCallback(verbose=2))

    # Checkpoint every 50k steps
    callbacks.append(CheckpointCallback(save_freq=50000, save_path=f'./models/{args.pair}', name_prefix=args.algo))

    # --- NEW: EVAL CALLBACK ---
    # This pauses training every 20,000 steps to test on 2024 data
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{args.pair}_best_eval',
        log_path=f'./logs/{args.pair}_eval',
        eval_freq=20000,   # How often to test (in steps)
        n_eval_episodes=1, # Test 1 full run of 2024
        deterministic=True, # Use "best action" (no random exploring) during test
        render=False
    )
    callbacks.append(eval_callback)

    # --- 5. TRAIN ---
    print(f"\n--- 3. STARTING TRAINING ({args.total_timesteps} steps) ---")
    print(f"    (Evaluation on 2024 data will occur every 20,000 steps)")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    
    model.save(f"{args.algo}_{args.pair}_final")
    print("Training complete.")

if __name__ == "__main__":
    main()
    