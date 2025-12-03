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

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.vec_env")
warnings.simplefilter(action='ignore', category=FutureWarning)

# SB3 Imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# WandB
import wandb
from wandb.integration.sb3 import WandbCallback

# Local Imports
from enhanced_trading_env import EnhancedTradingEnv
from callbacks.base_callbacks import SaveOnBestTrainingRewardCallback, TensorboardCallback, CustomEvalCallback
from callbacks.feature_saliency import FeatureSaliencyCallback
from fetch_metrics import generate_metrics

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.vec_env")
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments (Threads)")
    
    # Thresholds
    parser.add_argument("--buy-threshold", type=float, default=0.5, help="Threshold to Buy (> X)")
    parser.add_argument("--sell-threshold", type=float, default=-0.5, help="Threshold to Sell (< X)")
    
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
            sync_tensorboard=True, 
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
    
    # 1. Calculate VP once in the main process
    from volume_profile import get_rolling_vp
    print("Pre-calculating Volume Profile...")
    vp_data = {}
    for d in args.vp_days:
        vp_data[d] = get_rolling_vp(train_df, d, bins=args.vp_bins)

    # 3. Create Vectorized Environment (Training)
    env_kwargs = {
        'initial_balance': args.initial_balance,
        'vp_days': args.vp_days,
        'vp_bins': args.vp_bins,
        'buy_threshold': args.buy_threshold,
        'sell_threshold': args.sell_threshold,
        'precalculated_vp': vp_data,  # Pass it here
        'trading_fee_multiplier': 0.002,  # 0.2% fee (Simulates slippage + fee)
    }
    
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs['df'] = train_df
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    
    env = make_vec_env(
        EnhancedTradingEnv, 
        n_envs=args.n_envs, 
        env_kwargs=train_env_kwargs,
        vec_env_cls=vec_env_cls,
        monitor_dir="./logs/"
    )
    
    # 4. Create Evaluation Environment
    eval_env = None
    if test_df is not None and not test_df.empty:
        print("Pre-calculating Volume Profile for eval...")
        vp_data_test = {}
        for d in args.vp_days:
            vp_data_test[d] = get_rolling_vp(test_df, d, bins=args.vp_bins)
        print("Creating Evaluation environment...")
        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs['precalculated_vp'] = vp_data_test
        eval_env_kwargs['df'] = test_df
        eval_env = make_vec_env(EnhancedTradingEnv, n_envs=1, env_kwargs=eval_env_kwargs)
    else:
        print("⚠️ WARNING: Test dataset is EMPTY. Evaluation charts will remain blank.")
        print(f"   Please check your --test-split date ({args.test_split}) vs your CSV data range.")

    # 5. Create Single Instance for Saliency
    dummy_saliency_env = EnhancedTradingEnv(**train_env_kwargs)

    # 6. Model Setup
    policy_kwargs = dict(net_arch=[512, 512, 512]) if args.algo == 'sac' else dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    
    tensorboard_log = f"./{args.algo}_tb/"
    model = None
    
    # --- RESUME LOGIC ---
    if args.resume:
        load_path = args.model_path
        if load_path is None:
            # 1. Try Best Model
            default_best = f"logs/best_model.zip"
            if os.path.exists(default_best):
                load_path = default_best
            # 2. Try Latest Checkpoint
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
    else:
        # delete directory for {algo}_tb
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)  

        # remove also checkpoints for this algo/pair to avoid confusion
        chk_dir = f"checkpoints/{args.algo}_{args.pair}"
        if os.path.exists(chk_dir):
            shutil.rmtree(chk_dir)

        # remove also models for this models/{args.algo}_{args.pair}*.* using glob

        model_files = glob.glob(f"models/{args.algo}_{args.pair}*.*")
        for file in model_files:
            os.remove(file)
        #remove /{algo}_tb directory if it exists
        tb_dir = f"{args.algo}_tb"
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)

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
    
    # --- RESTORED SYSTEM INFO BLOCK ---
    sysinfo = get_system_info()
    print("System Information:")
    if torch.cuda.is_available():
        sysinfo[0]['GPU'] = torch.cuda.get_device_name()
    for key, value in sorted(sysinfo[0].items()):
        print(f"- {key}: {value}")
    # ----------------------------------

    # 7. Callbacks Setup
    
    # A. Checkpoint (Saves every 50k steps)
    checkpoint_path = f"checkpoints/{args.algo}_{args.pair}"
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=checkpoint_path,
        name_prefix=f"{args.algo}_vp{args.vp_bins}"
    )

    # B. Best Training Reward (Saved separately)
    train_reward_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir="./logs/")
    train_reward_callback.save_path = os.path.join("./logs/", "best_training_model")

    callbacks = [
        TensorboardCallback(),
        FeatureSaliencyCallback(dummy_saliency_env, check_freq=50000),
        checkpoint_callback,
        train_reward_callback
    ]

    # C. Evaluation Callback
    if eval_env is not None:
        print("✅ EvalCallback attached. Validation will run every 20,000 steps.")
        print("   -> Best performing model on TEST data will be saved as 'logs/best_model.zip'")
        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=20000,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
    
    if args.wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=20000,
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
        model.save(f"models/{args.algo}_{args.pair}_interrupted")
    # 9. Save Final
    final_path = f"models/{args.algo}_{args.pair}_final"
    model.save(final_path)
    print(f"✅ Model saved to {final_path}")

    if args.wandb:
        generate_metrics()
        wandb.finish()

if __name__ == "__main__":
    main()