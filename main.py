import os
import argparse
import pandas as pd
import numpy as np
import torch
import warnings
import sys
import shutil
import glob
from stable_baselines3.common.utils import get_system_info

# Fixes "The behavior of DataFrame concatenation..."
warnings.simplefilter(action='ignore', category=FutureWarning)

# SB3 Imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# ADDED EvalCallback here
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# WandB
import wandb
from wandb.integration.sb3 import WandbCallback

# Local Imports
from trading_env import TradingEnv
from callbacks.base_callbacks import SaveOnBestTrainingRewardCallback, TensorboardCallback
from callbacks.feature_saliency import FeatureSaliencyCallback

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Trading Bot")
    
    # Data & Config
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--data-path", type=str, default="data/BTCUSDT_1h.csv", help="Path to CSV data")
    
    # Environment
    parser.add_argument("--vp-days", type=int, nargs='+', default=[7, 30], help="Volume Profile days (e.g. 7 30)")
    parser.add_argument("--vp-bins", type=int, default=40, help="Number of bins for VP Heatmap")
    parser.add_argument("--initial-balance", type=float, default=1000, help="Starting cash")
    parser.add_argument("--n-envs", type=int, default=14, help="Number of parallel environments (Threads)")
    
    # Training
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="RL Algorithm")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning Rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor (High for long-term)")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    
    # Validation & Resume
    parser.add_argument("--test-split", type=str, default=None, help="Date to split Train/Test (e.g. '2023-01-01')")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to specific model to load")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--project-name", type=str, default="ai-trading-bot", help="WandB Project Name")
    
    return parser.parse_args()

def load_and_process_data(csv_path):
    if not os.path.exists(csv_path):
        if os.path.exists("BTCUSDT_data.csv"):
            csv_path = "BTCUSDT_data.csv"
        else:
            raise FileNotFoundError(f"Data not found at {csv_path}")
            
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def main():
    args = parse_args()
    
    # 1. Initialize WandB
    if args.wandb:
        run_name = f"{args.algo}_{args.pair}_VP{args.vp_bins}_Envs{args.n_envs}"
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=run_name,
            sync_tensorboard=True, # Handles steps automatically
            monitor_gym=True
        )
    
    # 2. Load Data
    df = load_and_process_data(args.data_path)
    
    train_df = df.copy()
    test_df = None
    
    if args.test_split:
        mask = df['date'] < args.test_split
        train_df = df[mask].reset_index(drop=True)
        test_df = df[~mask].reset_index(drop=True)
        print(f"Split Data: Train ({len(train_df)}) | Test ({len(test_df)})")
    
    # 3. Create Vectorized Environment (Training)
    env_kwargs = {
        'initial_balance': args.initial_balance,
        'vp_days': args.vp_days,
        'vp_bins': args.vp_bins
    }
    
    # Copy kwargs and add the specific dataframe
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs['df'] = train_df
    
    print(f"Creating {args.n_envs} parallel environments for Training...")
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    
    env = make_vec_env(
        TradingEnv, 
        n_envs=args.n_envs, 
        env_kwargs=train_env_kwargs,
        vec_env_cls=vec_env_cls,
        monitor_dir="./logs/"
    )
    
    # 4. Create Evaluation Environment (Single Thread)
    eval_env = None
    if test_df is not None and not test_df.empty:
        print("Creating Evaluation environment...")
        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs['df'] = test_df
        # Eval env using same vec_env_cls as training for consistency
        eval_env = make_vec_env(TradingEnv, n_envs=1, env_kwargs=eval_env_kwargs, vec_env_cls=vec_env_cls)

    # 5. Create Single Instance for Saliency
    dummy_saliency_env = TradingEnv(**train_env_kwargs)

    # 6. Model Setup
    policy_kwargs = dict(net_arch=[512, 512, 512]) if args.algo == 'sac' else dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    
    tensorboard_log = f"./{args.algo}_tb/"
    model = None
    
    # --- RESUME LOGIC ---
    if args.resume:
        load_path = args.model_path
        if load_path is None:
            default_best = f"logs/best_model.zip"
            if os.path.exists(default_best):
                load_path = default_best
            else:
                chk_dir = f"checkpoints/{args.algo}_{args.pair}"
                if os.path.exists(chk_dir):
                    files = glob.glob(f"{chk_dir}/*.zip")
                    if files:
                        load_path = max(files, key=os.path.getctime)
        
        if load_path and os.path.exists(load_path):
            print(f"📥 RESUMING: Loading model from {load_path}")
            if args.algo == 'sac':
                model = SAC.load(load_path, env=env, print_system_info=True, device=args.device, tensorboard_log=tensorboard_log)
            else:
                model = PPO.load(load_path, env=env, print_system_info=True, device=args.device, tensorboard_log=tensorboard_log)
        else:
             print("⚠️ Resume requested but no model found. Starting fresh.")

    # --- FRESH START LOGIC ---
    if model is None:
        if not args.resume:
            if os.path.exists(tensorboard_log):
                shutil.rmtree(tensorboard_log)

        print(f"✨ Creating NEW {args.algo.upper()} Model")
        if args.algo == 'sac':
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                buffer_size=100_000,
                batch_size=args.batch_size,
                ent_coef='auto',
                gamma=args.gamma,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=tensorboard_log,
                device=args.device
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                n_steps=2048,
                batch_size=args.batch_size,
                n_epochs=10,
                gamma=args.gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=tensorboard_log,
                device=args.device
            )
    else:
        # delete directory for {algo}_tb
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)  

        # remove also checkpoints for this algo/pair to avoid confusion
        chk_dir = f"checkpoints/{args.algo}_{args.pair}"
        if os.path.exists(chk_dir):
            shutil.rmtree(chk_dir)

        # remove also models for this algo/pair to avoid confusion
        model_dir = f"models/{args.algo}_{args.pair}"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)        


        
    sysinfo = get_system_info()
    print("System Information:")
    if torch.cuda.is_available():
        sysinfo[0]['GPU'] = torch.cuda.get_device_name()
    for key, value in sorted(sysinfo[0].items()):
        print(f"- {key}: {value}")

    # 7. Callbacks
    
    # Checkpoint Callback
    checkpoint_path = f"checkpoints/{args.algo}_{args.pair}"
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=checkpoint_path,
        name_prefix=f"{args.algo}_vp{args.vp_bins}"
    )

    callbacks = [
        TensorboardCallback(),
        FeatureSaliencyCallback(dummy_saliency_env, check_freq=50000),
        checkpoint_callback
    ]

    # --- ADDED: Evaluation Callback (Fixes empty Evaluation charts) ---
    if eval_env is not None:
        print("✅ EvalCallback attached. Validation will run every 20,000 steps.")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=20000, 
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
    else:
        # Fallback if no test split provided
        callbacks.append(SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir="./logs/"))
    
    if args.wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"models/{args.algo}_{args.pair}",
            verbose=2
        ))
        
    callback_list = CallbackList(callbacks)
    
    # 8. Learn
    print(f"🚀 Starting Training for {args.total_timesteps} steps...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback_list, progress_bar=True)
    except KeyboardInterrupt:
        print("🛑 Training interrupted. Saving...")
        
    # 9. Save
    final_path = f"models/{args.algo}_{args.pair}_final"
    model.save(final_path)
    print(f"✅ Model saved to {final_path}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
