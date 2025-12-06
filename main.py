import os
import argparse
import glob
import shutil
import pandas as pd
import numpy as np
import torch
import time
import datetime
import subprocess

# RL & Gym
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C, TD3
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from sb3_contrib.common.torch_layers import TransformerNet
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

# WandB
import wandb
from wandb.integration.sb3 import WandbCallback

# Custom Modules
from enhanced_trading_env import EnhancedTradingEnv
from callbacks.base_callbacks import TensorboardCallback, CustomEvalCallback
from callbacks.feature_saliency import FeatureSaliencyCallback
from callbacks.recurrent_saliency import RecurrentFeatureSaliencyCallback
class HybridLstmTransformerPolicy(MlpLstmPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, lstm_hidden_size=128, **kwargs):

        lstm_layers = kwargs.pop('lstm_layers', 2)

        n_heads = kwargs.pop('n_heads', 2)

        super().__init__(observation_space, action_space, lr_schedule, lstm_hidden_size=lstm_hidden_size, **kwargs)

        # Modify lstm to have layers

        self.lstm = nn.LSTM(self.features_dim, lstm_hidden_size, lstm_layers, batch_first=True)

        # Add transformer

        self.transformer = TransformerNet(lstm_hidden_size, 2, n_heads, 256)
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
    parser.add_argument("--buy-threshold", type=float, default=0.4, help="Threshold to trigger buy action")
    parser.add_argument("--sell-threshold", type=float, default=-0.4, help="Threshold to trigger sell action")
    
    # Environment Config
    # REVERTED TO YOUR DEFAULT: [7, 30]
    parser.add_argument("--n-heads", type=int, default=2, help="Number of heads for transformer")
    parser.add_argument("--vp-days", type=int, nargs='+', default=[7, 30], help="Volume Profile days (e.g. 7 30)")
    parser.add_argument("--vp-bins", type=int, default=40, help="Volume Profile bins")
    parser.add_argument("--window-size", type=int, default=50, help="Observation window size")
    parser.add_argument("--n-envs", type=int, default=15, help="Number of parallel environments")
    parser.add_argument("--phase", type=int, default=1, help="Curriculum phase (1=profit, 2=sortino, 3=mdd)")
    
    # RL Config
    parser.add_argument("--algo", type=str, default="recurrentppo", choices=["sac", "ppo", "a2c", "td3", "recurrentppo"], help="RL Algorithm")
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
    print("Starting main function...")
    args = parse_args()
    print(f"Parsed args: {args}")
    set_random_seed(args.seed)
    print("Random seed set.")
    
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
    print("Calculating VP for training data...")
    vp_data_train = {}
    for days in args.vp_days:
        print(f"Calculating Rolling VP for {days} days (Bins: {args.vp_bins})...")
        vp_data_train[days] = get_rolling_vp(train_df, days, bins=args.vp_bins)
    print("Train VP calculation complete.")

    # 2. Test VP
    print("Calculating VP for test data...")
    vp_data_test = {}
    for days in args.vp_days:
        vp_data_test[days] = get_rolling_vp(test_df, days, bins=args.vp_bins)
    print("Test VP calculation complete.")

    # --- Environment Setup ---
    print("Setting up environments...")
    # Force window_size=1 for RecurrentPPO to avoid confusion with LSTM memory
    window_size = 1 if args.algo.lower() == 'recurrentppo' else args.window_size
    env_kwargs = {
        'initial_balance': args.initial_balance,
        'vp_days': args.vp_days,
        'vp_bins': args.vp_bins,
        'lookback_window': window_size,
        'buy_threshold': args.buy_threshold,
        'sell_threshold': args.sell_threshold,
        'trading_fee_multiplier': args.trading_fee,
        'phase': args.phase  # New: Pass phase
    }
    print(f"Env kwargs: {env_kwargs}")

    # Training Env
    print("Creating training environment...")
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs['df'] = train_df
    train_env_kwargs['precalculated_vp'] = vp_data_train

    train_env = make_vec_env(
        EnhancedTradingEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=train_env_kwargs
    )
    print("Training environment created.")

    # --- Reactivate Normalization for Training ---
    print("Applying VecNormalize to training env...")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    print("VecNormalize applied.")

    # No VecFrameStack needed for RecurrentPPO - LSTM handles temporal dependencies internally

    # Evaluation Env (Raw for accurate metrics)
    print("Creating evaluation environment...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['df'] = test_df
    eval_env_kwargs['precalculated_vp'] = vp_data_test

    eval_env = DummyVecEnv([lambda: EnhancedTradingEnv(**eval_env_kwargs)])
    print("Evaluation environment created.")

    # No VecFrameStack needed for RecurrentPPO - LSTM handles temporal dependencies internally
    # Optional: Load train stats for eval (set training=False)
    # eval_env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", eval_env); eval_env.training = False

    # Create dummy env for saliency callback (skip for RecurrentPPO due to LSTM compatibility issues)
    saliency_callback = None
    if args.algo.lower() != 'recurrentppo':
        dummy_env = eval_env.envs[0]
        saliency_callback = FeatureSaliencyCallback(dummy_env=dummy_env, check_freq=50000)

    # --- W&B Setup ---
    print("Setting up W&B..." if args.wandb else "Skipping W&B setup.")
    if args.wandb:
        # Get git branch name
        try:
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
        except:
            git_branch = "unknown"

        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Determine a Stable Name (No Timestamp)
        # This groups runs of the same config together visually
        run_name = f"{git_branch}_{args.algo}_{args.pair}_VP{args.vp_bins}_Envs{args.n_envs}"

        # 2. Handle Resuming Logic (The "Pro" Fix)
        # We save the Run ID to a file. If we resume, we load it back.
        # This stitches the charts together seamlessly.
        id_file_path = f"logs/.wandb_id_{args.algo}_{args.pair}.txt"
        os.makedirs("logs", exist_ok=True)

        run_id = None
        if args.resume and os.path.exists(id_file_path):
            with open(id_file_path, "r") as f:
                run_id = f.read().strip()
            print(f"🔄 Resuming W&B Run ID: {run_id}")
        elif not args.resume:
            # If starting fresh, generate a new ID and save it
            run_id = wandb.util.generate_id()
            with open(id_file_path, "w") as f:
                f.write(run_id)

        wandb.init(
            project="ai-trading-bot",
            entity="zhivko",
            config=vars(args),
            name=run_name,   # Stable name
            id=run_id,       # Force specific ID to stitch charts
            resume="allow",  # Allow resuming if ID exists
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True
        )
        print(f"W&B initialized (Run: {run_name})")

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=f"./checkpoints/{args.algo}_{args.pair}", 
        name_prefix=args.algo
    )
    
    tensorboard_callback = TensorboardCallback(verbose=1, buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)

    callbacks = [tensorboard_callback, checkpoint_callback]
    if saliency_callback is not None:
        callbacks.append(saliency_callback)
    
    if args.wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=0,
            model_save_path=f"models/{args.algo}_{args.pair}_wb",
            verbose=0  # Reduced from 2 to minimize logging overhead
        ))

    eval_freq_adjusted = max(50000 // args.n_envs, 1) # e.g., 50000 // 16 = 3125 calls

    # Add after eval_env setup
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq_adjusted,  # Reduced from 10k to 50k steps to reduce overhead
        log_path="./logs/",
        best_model_save_path="./models/",
        deterministic=True,
        render=False,
        test_split=args.test_split,
        pair=args.pair,
        initial_balance=args.initial_balance
    )
    callbacks.append(eval_callback)

    # Add Recurrent Saliency Callback for RecurrentPPO
    if args.algo.lower() == 'recurrentppo':
        # -----------------------------------------------------
        # ROBUST FEATURE NAME RETRIEVAL
        # -----------------------------------------------------
        try:
            # Method A: Try calling the function directly (Best for SubprocVecEnv)
            feature_names = train_env.env_method("get_feature_names", indices=0)[0]
        except Exception as e:
            print(f"Warning: env_method failed ({e}). Trying attribute access...")
            try:
                # Method B: Try accessing the attribute
                feature_names = train_env.get_attr("feature_names", indices=0)[0]
            except Exception as e2:
                # Method C: Fallback
                print(f"Warning: Could not retrieve feature names ({e2}). Using generic labels.")
                obs_dim = train_env.observation_space.shape[0]
                feature_names = [f"F_{i}" for i in range(obs_dim)]

        # Initialize the callback
        saliency_cb = RecurrentFeatureSaliencyCallback(
            check_freq=10000,           # Check every 10k steps
            save_path="./logs/saliency",
            feature_names=feature_names,
            verbose=1
        )
        callbacks.append(saliency_cb)

    callback_list = CallbackList(callbacks)

    # --- Model Initialization ---
    AlgoClass = ALGO_MAP.get(args.algo.lower())
    if not AlgoClass:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Hyperparameters - Algorithm-specific
    policy = "MlpPolicy"
    if args.algo.lower() == 'recurrentppo':
        policy = "HybridLstmTransformerPolicy"
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=128,
            lstm_layers=2,
            n_heads=args.n_heads
        )
    elif args.algo.lower() in ['sac', 'td3']:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    # Resume Logic
    model_path = f"./models/{args.algo}_{args.pair}"
    best_model_path = f"{log_dir}/best_model.zip"
    norm_path = f"{log_dir}/vec_normalize.pkl"
    
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
            print(f"RESUMING training from: {load_path}")
            try:
                # Load model
                model = AlgoClass.load(load_path, env=train_env, device=args.device, tensorboard_log=tensorboard_log)
                # Sync VecNormalize stats
                if os.path.exists(f"{load_path}.pkl"):
                    train_env = VecNormalize.load(f"{load_path}.pkl", train_env)
                    train_env.training = True
                # CRITICAL: Do not reset steps when resuming
                reset_num_timesteps = False
                print(f"   > Resuming from Global Step: {model.num_timesteps}")
            except ValueError as e:
                if "Observation spaces do not match" in str(e):
                    print(f"Model incompatible due to env changes: {e}")
                    print("Starting fresh training.")
                    model = None
                    reset_num_timesteps = True
                    # Clean old logs if incompatible
                    if os.path.exists(tensorboard_log):
                        shutil.rmtree(tensorboard_log)
                else:
                    raise
        else:
            print("Resume requested but no model found. Starting FRESH.")
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
        print(f"Initializing new {args.algo.upper()} model...")

        # Clean old logs only if starting fresh
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)

        # Clean old checkpoints
        chk_dir = f"checkpoints/{args.algo}_{args.pair}"
        if os.path.exists(chk_dir):
            shutil.rmtree(chk_dir)

        # Set model hyperparameters
        model_kwargs = {}
        if args.algo.lower() == 'recurrentppo':
            model_kwargs.update(dict(
                n_steps=1024,
                batch_size=16384,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            ))
        else:
            model_kwargs.update(dict(
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            ))

        if args.algo.lower() == 'sac':
            model_kwargs['buffer_size'] = 100000
            model_kwargs['ent_coef'] = 'auto'

        model = AlgoClass(
            "HybridLstmTransformerPolicy",
            train_env,
            verbose=1,
            tensorboard_log=f"./logs/{args.algo}_tensorboard",
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.batch_size,  # Ensure n_steps is aligned with batch logic if needed
            seed=args.seed,
            device=args.device,
            policy_kwargs=policy_kwargs
        )
        reset_num_timesteps = True

    # --- Train ---
    print(f"Training started... Target: {args.total_timesteps} steps")
    print(f"Model: {args.algo.upper()}, Device: {args.device}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callback_list, 
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps # <--- Handles the resumption of step count
        )
        
        # Save Final Model + Normalize
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        model.save(model_path)
        train_env.save(f"{model_path}.pkl")
        print(f"Training Complete. Model saved to {model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving model...")
        model.save(model_path)
        train_env.save(f"{model_path}.pkl")

if __name__ == "__main__":
    main()