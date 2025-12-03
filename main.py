import os
import argparse
import glob
import shutil
import pandas as pd
import numpy as np
import torch
import time

# RL & Gym
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

# WandB
import wandb
from wandb.integration.sb3 import WandbCallback

# Custom Modules
from enhanced_trading_env import EnhancedTradingEnv
from callbacks.base_callbacks import TensorboardCallback

from callbacks.base_callbacks import CustomEvalCallback

from volume_profile import get_rolling_vp

# --- Custom Callback for Train Reward Logging ---
class TrainRewardCallback(BaseCallback):
    def __init__(self, check_freq):
        super(TrainRewardCallback, self).__init__(verbose=1)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward (approximation)
            # 'rollout/ep_rew_mean' is usually available in self.logger.name_to_value
            # But direct access via locals is harder in SB3.
            # We rely on standard logging.
            pass
        return True

# ---------------------------------------------------------
# 1. Configuration & Arguments
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RL Trading Bot - Main Trainer")
    
    # Trading Config
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Data timeframe")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Starting money")
    parser.add_argument("--trading-fee", type=float, default=0.00075, help="Trading fee (0.075% default)")
    parser.add_argument("--buy-threshold", type=float, default=0.1, help="Threshold to trigger buy action")
    parser.add_argument("--sell-threshold", type=float, default=-0.1, help="Threshold to trigger sell action")
    
    # Environment Config
    # REVERTED TO YOUR DEFAULT: [7, 30]
    parser.add_argument("--vp-days", type=int, nargs='+', default=[7, 30], help="Volume Profile days (e.g. 7 30)")
    parser.add_argument("--vp-bins", type=int, default=40, help="Volume Profile bins")
    parser.add_argument("--window-size", type=int, default=50, help="Observation window size")
    parser.add_argument("--n-envs", type=int, default=15, help="Number of parallel environments")
    
    # RL Config
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo", "a2c", "td3", "recurrentppo"], help="RL Algorithm")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    
    # System Config
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--test-split", type=str, default="2023-01-01", help="Date to split Train/Test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()

# Algorithm Mapping
ALGO_MAP = {
    "sac": SAC,
    "ppo": PPO,
    "a2c": A2C,
    "td3": TD3,
    "recurrentppo": RecurrentPPO
}

# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------
def load_and_process_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure standard columns
    df.columns = [c.lower() for c in df.columns]
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ema_50']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
            
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Minimal preprocessing before passing to Env (Env handles indicators)
    # But we ensure no NaNs and sort by date
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# ---------------------------------------------------------
# 3. Main Execution Flow
# ---------------------------------------------------------
def main():
    args = parse_args()
    set_random_seed(args.seed)
    
    # Paths
    data_file = f"{args.pair}_data.csv"
    log_dir = "./logs/"
    tensorboard_log = f"./{args.algo}_tb/"
    
    # --- Load Data ---
    df = load_and_process_data(data_file)
    
    # Split Train/Test
    split_idx = df[df['timestamp'] >= args.test_split].index[0]
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Split Data: Train ({len(train_df)}) | Test ({len(test_df)})")

    # --- Pre-Calculate Volume Profile (Multiprocessing Fix) ---
    print(f"Creating {args.n_envs} parallel environments...")
    
    print(f"--- Initializing EnhancedTradingEnv (Target Bins: {args.vp_bins}) ---")
    
    # 1. Train VP
    vp_data_train = {}
    for days in args.vp_days:
        print(f"⚙️ [VP] Calculating Rolling VP for {days} days (Bins: {args.vp_bins})...")
        vp_data_train[days] = get_rolling_vp(train_df, days, bins=args.vp_bins)

    # 2. Test VP
    vp_data_test = {}
    for days in args.vp_days:
        vp_data_test[days] = get_rolling_vp(test_df, days, bins=args.vp_bins)

    # --- Environment Setup ---
    env_kwargs = {
        'initial_balance': args.initial_balance,
        'vp_days': args.vp_days,
        'vp_bins': args.vp_bins,
        'lookback_window': args.window_size,
        'buy_threshold': args.buy_threshold,
        'sell_threshold': args.sell_threshold,
        'trading_fee_multiplier': args.trading_fee
    }

    # Training Env
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs['df'] = train_df
    train_env_kwargs['precalculated_vp'] = vp_data_train

    env = make_vec_env(
        EnhancedTradingEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=train_env_kwargs
    )
    
    # --- CRITICAL FIX: DISABLED NORMALIZATION ---
    # We commented this out to ensure the model learns on RAW data.
    # This makes the model compatible with the Backtester which uses raw data.
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Evaluation Env (Dummy for Single Process)
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['df'] = test_df
    eval_env_kwargs['precalculated_vp'] = vp_data_test
    
    eval_env = DummyVecEnv([lambda: EnhancedTradingEnv(**eval_env_kwargs)])
    # For Eval, we usually wrap in VecNormalize but set training=False to use stats from train env
    # However, keeping it simple here for now or syncing stats manually later.
    
    # --- W&B Setup ---
    if args.wandb:
        wandb.init(
            project="ai-trading-bot",
            entity="zhivko",
            config=vars(args),
            name=f"{args.algo}_{args.pair}_VP{args.vp_bins}_Envs{args.n_envs}",
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True
        )

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=f"./checkpoints/{args.algo}_{args.pair}", 
        name_prefix=args.algo
    )
    
    tensorboard_callback = TensorboardCallback(verbose=1)
    
    callbacks = [tensorboard_callback, checkpoint_callback]
    
    if args.wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=0, 
            model_save_path=f"models/{args.algo}_{args.pair}_wb",
            verbose=2
        ))

    # Add after eval_env setup
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=10000,  # Evaluate every 10k steps
        log_path="./logs/",
        best_model_save_path="./models/",
        deterministic=True,
        render=False,
        test_split=args.test_split,
        pair=args.pair,
        initial_balance=args.initial_balance
    )
    callbacks.append(eval_callback)

    callback_list = CallbackList(callbacks)

    # --- Model Initialization ---
    AlgoClass = ALGO_MAP.get(args.algo.lower())
    if not AlgoClass:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Hyperparameters
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    
    # Resume Logic
    model_path = f"./models/{args.algo}_{args.pair}"
    best_model_path = f"{log_dir}/best_model.zip"
    
    model = None
    reset_num_timesteps = True

    if args.resume:
        # 1. Try Best Model
        if os.path.exists(best_model_path):
            load_path = best_model_path
        # 2. Try Last Model
        elif os.path.exists(model_path + ".zip"):
            load_path = model_path + ".zip"
        # 3. Try Checkpoints
        else:
            chk_files = glob.glob(f"./checkpoints/{args.algo}_{args.pair}/*.zip")
            if chk_files:
                load_path = max(chk_files, key=os.path.getctime)
            else:
                load_path = None
        
        if load_path:
            print(f"♻️  RESUMING training from: {load_path}")
            # Pass tensorboard_log to continue writing to same graph
            model = AlgoClass.load(load_path, env=env, device=args.device, tensorboard_log=tensorboard_log)
            
            # CRITICAL: Do not reset steps when resuming
            reset_num_timesteps = False
            print(f"   > Resuming from Global Step: {model.num_timesteps}")
        else:
            print("⚠️  Resume requested but no model found. Starting FRESH.")
            # Clean logs if we failed to find a model to resume
            if os.path.exists(tensorboard_log):
                shutil.rmtree(tensorboard_log)
    else:
        # Delete all models for algo
        model_pattern = f"./models/{args.algo}_*.zip"
        model_files = glob.glob(model_pattern)
        for f in model_files:
            os.remove(f)

        # delete checkpoints for algo
        chk_pattern = f"./checkpoints/{args.algo}_*/**/*.zip"
        chk_files = glob.glob(chk_pattern, recursive=True)
        for f in chk_files:
            os.remove(f)    

        # delete tensorboard logs for algo
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)

    if model is None:
        print(f"🆕  Initializing new {args.algo.upper()} model...")
        
        # Clean old logs only if starting fresh
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)
            
        # Clean old checkpoints
        chk_dir = f"checkpoints/{args.algo}_{args.pair}"
        if os.path.exists(chk_dir):
            shutil.rmtree(chk_dir)

        model = AlgoClass(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            policy_kwargs=policy_kwargs,
            buffer_size=100000 if args.algo == 'sac' else None, # SAC specific
            ent_coef='auto' if args.algo == 'sac' else None,    # SAC specific
        )
        reset_num_timesteps = True

    # --- Train ---
    print(f"🚀  Training started... Target: {args.total_timesteps} steps")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callback_list, 
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps # <--- Handles the resumption of step count
        )
        
        # Save Final Model
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        model.save(model_path, metadata={"test_split": args.test_split, "pair": args.pair, "initial_balance": args.initial_balance})
        print(f"✅  Training Complete. Model saved to {model_path}")

    except KeyboardInterrupt:
        print("\n🛑  Training interrupted manually. Saving model...")
        model.save(model_path, metadata={"test_split": args.test_split, "pair": args.pair, "initial_balance": args.initial_balance})
        print("    Model saved.")

    finally:
        env.close()
        if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    main()