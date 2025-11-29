import argparse
import pandas as pd
import os
import numpy as np
import warnings
import torch
import glob # <--- Needed to find files

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

class WandbEvalListener(BaseCallback):
    """
    This callback runs AFTER the EvalCallback finishes testing.
    It grabs the results and logs them directly to WandB.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if this is called after evaluation
        if hasattr(self, 'locals') and self.locals and 'episode_reward' in self.locals:
            mean_reward = self.locals['episode_reward']
            mean_len = self.locals.get('episode_length', 0)

            print(f"📈 Sending Eval Metrics to WandB: {mean_reward}")

            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/mean_ep_length": mean_len,
                "global_step": self.num_timesteps
            })
        return True

# --- REAL-TIME CALLBACK ---
class RealTimeWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RealTimeWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if wandb.run is None: return True
        infos = self.locals["infos"][0]
        
        if self.num_timesteps % 100 == 0:
            current_lr = 0.0
            try: current_lr = self.model.policy.optimizer.param_groups[0]["lr"]
            except: pass

            current_alpha = 0.0
            if hasattr(self.model, "log_ent_coef"):
                try: current_alpha = np.exp(self.model.log_ent_coef.detach().cpu().item())
                except: pass

            wandb.log({
                "realtime/portfolio_value": infos.get("portfolio_value", 0),
                "realtime/balance": infos.get("balance", 0),
                "realtime/step_reward": infos.get("reward", 0),
                "realtime/action": infos.get("action", 0),
                "realtime/alpha_entropy": current_alpha,
                "realtime/learning_rate": current_lr,
                
                # --- CHECK THIS BLOCK ---
                "realtime/market_context": {
                    "realtime/price_main": infos.get("price", 0),
                    "realtime/price_poc": infos.get("poc", 0),  # <--- MUST match trading_env keys
                    "realtime/price_vah": infos.get("vah", 0),  # <--- MUST match trading_env keys
                    "realtime/price_val": infos.get("val", 0),  # <--- MUST match trading_env keys
                },
                # ------------------------
                
                "global_step": self.num_timesteps
            })
            
        if self.num_timesteps % 2000 == 0 and "vp_heatmap" in infos:
            heatmap = infos["vp_heatmap"]
            price_bins = infos.get("vp_bins", []) 
            current_price = infos.get("price", 0)

            if len(price_bins) == len(heatmap):
                data = []
                for p, vol in zip(price_bins, heatmap):
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
    parser.add_argument("--test-split", type=str, default="2024-01-01", help="Date to split Train/Test")
    
    # --- UPDATED RESUME ARGUMENT ---
    # nargs='?' means:
    # 1. No flag -> None
    # 2. --resume -> 'LATEST' (via const)
    # 3. --resume file.zip -> 'file.zip'
    parser.add_argument("--resume", nargs='?', const='LATEST', default=None, help="Resume training. Auto-loads latest if no file specified.")
    
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # --- 1. LOAD DATA ---
    csv_file = args.data_path if args.data_path else "BTCUSDT_data.csv"
    print(f"\n--- 1. LOADING DATA: {csv_file} ---")
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        if 'timestamp' in df.columns: df.rename(columns={'timestamp': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"✂️ Splitting data at {args.test_split}...")
        train_df = df[df.index < args.test_split].copy()
        test_df = df[df.index >= args.test_split].copy()
        
        print(f"   📘 Training: {len(train_df)} rows")
        print(f"   qh Testing:  {len(test_df)} rows")
    else:
        print("❌ Error: File not found.")
        return

    # --- 2. INIT ENVIRONMENTS ---
    print("\n--- 2. INITIALIZING ENVIRONMENTS ---")
    print("-> Training Env...")
    env = DummyVecEnv([lambda df=train_df: Monitor(TradingEnv(df, vp_days=args.vp_days))])
    
    print("-> Evaluation Env...")
    eval_env = DummyVecEnv([lambda df=test_df: Monitor(TradingEnv(df, vp_days=args.vp_days))])

    # --- 3. MODEL SETUP (Smart Resume) ---
    policy_kwargs = dict(net_arch=[256, 256])
    tensorboard_log = f"./{args.algo}_tb/"
    
    model_path_to_load = None
    
    # --- SMART RESUME LOGIC ---
    if args.resume is not None:
        if args.resume == 'LATEST':
            # Auto-detect latest file in models folder
            models_dir = f"./models/{args.pair}"
            if not os.path.exists(models_dir):
                raise FileNotFoundError(f"Cannot resume: Directory {models_dir} does not exist.")
            
            # Get all zip files
            list_of_files = glob.glob(f"{models_dir}/*.zip")
            if not list_of_files:
                raise FileNotFoundError(f"Cannot resume: No .zip files found in {models_dir}")
            
            # Find the one with the latest modification time
            latest_file = max(list_of_files, key=os.path.getmtime)
            model_path_to_load = latest_file
            print(f"\n🔄 Auto-detected latest checkpoint: {model_path_to_load}")
        else:
            # User provided a specific path
            model_path_to_load = args.resume
            print(f"\n🔄 Resuming from specified file: {model_path_to_load}")

    if model_path_to_load:
        if not os.path.exists(model_path_to_load):
            raise FileNotFoundError(f"Model file {model_path_to_load} not found!")

        if args.algo.lower() == 'sac':
            model = SAC.load(model_path_to_load, env=env, tensorboard_log=tensorboard_log)
        else:
            model = PPO.load(model_path_to_load, env=env, tensorboard_log=tensorboard_log)
        print("✅ Model weights loaded.")
    else:
        print(f"\n✨ INITIALIZING NEW MODEL ({args.algo.upper()})")
        if args.algo.lower() == 'sac':
            model = SAC(
                "MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log=tensorboard_log, 
                learning_rate=3e-4,
                ent_coef='auto'
            )
        else:
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log)

    # --- 4. CALLBACKS ---
    callbacks = []
    
    if args.wandb:
        wandb.init(
            project="ai-trading-bot", 
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            # If resuming, we usually start a new run to avoid messing up charts with gaps
        )
        callbacks.append(RealTimeWandbCallback())
        callbacks.append(WandbCallback(verbose=2))

    # Checkpoint logic
    callbacks.append(CheckpointCallback(save_freq=50000, save_path=f'./models/{args.pair}', name_prefix=args.algo))

    # Create the listener
    eval_listener = WandbEvalListener()

    # Eval logic
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{args.pair}_best_eval',
        log_path=f'./sac_tb/',
        eval_freq=20000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=eval_listener
    )
    callbacks.append(eval_callback)

    # --- 5. TRAIN ---
    print(f"\n--- 3. STARTING TRAINING ({args.total_timesteps} steps) ---")
    
    # reset_num_timesteps=False allows logging to continue where it left off
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=(model_path_to_load is None))
    
    model.save(f"{args.algo}_{args.pair}_final")
    print("Training complete.")


# usage
# python.exe c:/git/ai-tradig-bot-3/main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2024-01-01 --total-timesteps 1000000 --wandb
# or
# python.exe c:/git/ai-tradig-bot-3/main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2024-01-01 --total-timesteps 1000000 --wandb --resume
if __name__ == "__main__":
    main()