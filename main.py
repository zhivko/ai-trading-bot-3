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
import logging
import signal
import sys
import traceback
import threading

# Matplotlib backend fix for server environments
import matplotlib
matplotlib.use('Agg')

# RL & Gym
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

# WandB
import wandb
from wandb.integration.sb3 import WandbCallback

# Custom Modules
from enhanced_trading_env import EnhancedTradingEnv
from callbacks.base_callbacks import TensorboardCallback, CustomEvalCallback, ProgressBarCallback
from callbacks.recurrent_saliency import RecurrentFeatureSaliencyCallback
from volume_profile import get_rolling_vp

# --- Debug Signal Handler for Thread Stack Traces ---
def debug_signal_handler(signum, frame):
    """
    Catches Ctrl+C (SIGINT) and prints the stack trace of ALL running threads
    before exiting. This helps debug hangs by showing where each thread is stuck.
    """
    logging.info(f"\n\n!!! CAUGHT CTRL+C (Signal {signum}) !!!")
    logging.info("Dumping stack traces for all running threads to see where it hangs...\n")

    # Map thread IDs to their names for cleaner output
    id2name = {t.ident: t.name for t in threading.enumerate()}

    for thread_id, stack in sys._current_frames().items():
        name = id2name.get(thread_id, f"Thread ID {thread_id}")
        logging.info(f"--- Stack trace for Thread: {name} ---")
        stack_trace = ''.join(traceback.format_stack(stack))
        logging.info(stack_trace)
        logging.info("-" * 40 + "\n")

    logging.info("Exiting application...")
    sys.exit(1)

# ---------------------------------------------------------
# 1. Configuration & Arguments
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RL Trading Bot - Main Trainer")
    
    # Trading Config
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Data timeframe")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Starting money")
    parser.add_argument("--trading-fee", type=float, default=0.0015, help="Trading fee (0.15%)")
    parser.add_argument("--buy-threshold", type=float, default=0.4, help="Threshold to trigger buy action")
    parser.add_argument("--sell-threshold", type=float, default=-0.4, help="Threshold to trigger sell action")
    
    # Environment Config
    # REVERTED TO YOUR DEFAULT: [7, 30]
    parser.add_argument("--vp-days", type=int, nargs='+', default=[7, 30], help="Volume Profile days (e.g. 7 30)")
    parser.add_argument("--vp-bins", type=int, default=40, help="Volume Profile bins")
    parser.add_argument("--window-size", type=int, default=50, help="Observation window size")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--phase", type=int, default=1, help="Curriculum phase (1=profit, 2=sortino, 3=mdd)")
    
    # RL Config
    parser.add_argument("--algo", type=str, default="recurrentppo", choices=["sac", "ppo", "a2c", "td3", "recurrentppo"], help="RL Algorithm")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training steps")
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

    logging.info(f"Loading data from {filepath}...")
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
    # Delete old log file to start fresh
    if os.path.exists('ml.log'):
        os.remove('ml.log')
    logging.basicConfig(filename='ml.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s')
    logging.info("Starting main function...")
    args = parse_args()
    logging.info(f"Parsed args: {args}")
    set_random_seed(args.seed)
    logging.info("Random seed set.")
    
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
    
    logging.info(f"Split Data: Train ({len(train_df)}) | Test ({len(test_df)})")

    # --- Pre-Calculate Volume Profile (Multiprocessing Fix) ---
    logging.info(f"Creating {args.n_envs} parallel environments...")

    logging.info(f"--- Initializing EnhancedTradingEnv (Target Bins: {args.vp_bins}) ---")

    # 1. Train VP
    logging.info("Calculating VP for training data...")
    vp_data_train = {}
    for days in args.vp_days:
        logging.info(f"Calculating Rolling VP for {days} days (Bins: {args.vp_bins})...")
        vp_data_train[days] = get_rolling_vp(train_df, days, bins=args.vp_bins)
    logging.info("Train VP calculation complete.")

    # 2. Test VP
    logging.info("Calculating VP for test data...")
    vp_data_test = {}
    for days in args.vp_days:
        vp_data_test[days] = get_rolling_vp(test_df, days, bins=args.vp_bins)
    logging.info("Test VP calculation complete.")

    # --- Environment Setup ---
    logging.info("Setting up environments...")
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
    logging.info(f"Env kwargs: {env_kwargs}")

    # Training Env
    logging.info("Creating training environment...")
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
    logging.info("Training environment created.")

    # --- Reactivate Normalization for Training ---
    logging.info("Applying VecNormalize to training env...")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    logging.info("VecNormalize applied.")

    # No VecFrameStack needed for RecurrentPPO - LSTM handles temporal dependencies internally

    # Evaluation Env (Raw for accurate metrics)
    logging.info("Creating evaluation environment...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['df'] = test_df
    eval_env_kwargs['precalculated_vp'] = vp_data_test

    eval_env = DummyVecEnv([lambda: EnhancedTradingEnv(**eval_env_kwargs)])
    logging.info("Evaluation environment created.")

    # No VecFrameStack needed for RecurrentPPO - LSTM handles temporal dependencies internally
    # Optional: Load train stats for eval (set training=False)
    # eval_env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", eval_env); eval_env.training = False

    # Create dummy env for saliency callback (skip for RecurrentPPO due to LSTM compatibility issues)
    saliency_callback = None
    # if args.algo.lower() != 'recurrentppo':
    #     dummy_env = eval_env.envs[0]
    #     saliency_callback = FeatureSaliencyCallback(dummy_env=dummy_env, check_freq=10000)

    # --- W&B Setup ---
    logging.info("Setting up W&B..." if args.wandb else "Skipping W&B setup.")
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
            logging.info(f"🔄 Resuming W&B Run ID: {run_id}")
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
        logging.info(f"W&B initialized (Run: {run_name})")

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=f"./checkpoints/{args.algo}_{args.pair}", 
        name_prefix=args.algo
    )
    
    tensorboard_callback = TensorboardCallback(verbose=1, buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)

    progress_callback = ProgressBarCallback(update_interval=1000)
    callbacks = [progress_callback, tensorboard_callback, checkpoint_callback]
    if saliency_callback is not None:
        callbacks.append(saliency_callback)
    
    # if args.wandb:
    #     callbacks.append(WandbCallback(
    #         gradient_save_freq=0,
    #         model_save_path=f"models/{args.algo}_{args.pair}_wb",
    #         verbose=0  # Reduced from 2 to minimize logging overhead
    #     ))

    eval_freq_adjusted = max(50000 // args.n_envs, 1) # e.g., 50000 // 16 = 3125 calls

    # Add after eval_env setup
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq_adjusted,  # Reduced from 10k to 50k steps to reduce overhead
        log_path="./logs/",
        best_model_save_path="./models/",
        deterministic=True,
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
        feature_names = None
        try:
            # Method A: Try calling the function directly (Best for SubprocVecEnv & VecNormalize)
            # This allows us to reach through the wrappers to the actual Env code
            feature_names = train_env.env_method("get_feature_names", indices=0)[0]
            logging.info(f"✅ Successfully retrieved {len(feature_names)} feature names from environment.")
        except Exception as e:
            logging.warning(f"⚠️ env_method failed to get names ({e}). Trying attribute access...")
            try:
                # Method B: Try accessing the attribute (Backup)
                feature_names = train_env.get_attr("feature_names", indices=0)[0]
                logging.info(f"✅ Retrieved {len(feature_names)} feature names via attribute.")
            except Exception as e2:
                # Method C: Fallback to generics
                obs_dim = train_env.observation_space.shape[0]
                logging.warning(f"❌ Could not retrieve feature names ({e2}). Generating {obs_dim} generic labels.")
                feature_names = [f"F_{i}" for i in range(obs_dim)]

        # --- RE-ENABLE SALIENCY ---
        saliency_callback = RecurrentFeatureSaliencyCallback(
            check_freq=eval_freq_adjusted,  # Run every 50k steps (Heavy computation)
            save_path=os.path.join(log_dir, "saliency"),
            feature_names=feature_names,
            verbose=1
        )
        callbacks.append(saliency_callback)
        # --------------------------

    callback_list = CallbackList(callbacks)

    # --- Model Initialization ---
    AlgoClass = ALGO_MAP.get(args.algo.lower())
    if not AlgoClass:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Hyperparameters - Algorithm-specific
    policy = "MlpPolicy"
    # Define Network Architecture
    # SAC requires 'qf' (Q-Function), PPO requires 'vf' (Value Function)
    if args.algo.lower() == 'sac':
        net_arch = dict(pi=[512, 512], qf=[512, 512])
    else:
        net_arch = dict(pi=[512, 512], vf=[512, 512])

    policy_kwargs = dict(net_arch=net_arch)

    if args.algo.lower() == 'recurrentppo':
        policy = "MlpLstmPolicy"
        policy_kwargs.update(dict(
            ortho_init=False,  # Avoid LAPACK requirement for orthogonal init
            lstm_hidden_size=256,
            n_lstm_layers=2,
            shared_lstm=False,
            enable_critic_lstm=True,
            lstm_kwargs=dict(dropout=0.0)
        ))
    
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
            logging.info(f"RESUMING training from: {load_path}")
            try:
                # Load model
                model = AlgoClass.load(load_path, env=train_env, device=args.device, tensorboard_log=tensorboard_log)
                # Sync VecNormalize stats
                if os.path.exists(f"{load_path}.pkl"):
                    train_env = VecNormalize.load(f"{load_path}.pkl", train_env)
                    train_env.training = True
                # CRITICAL: Do not reset steps when resuming
                reset_num_timesteps = False
                logging.info(f"   > Resuming from Global Step: {model.num_timesteps}")
            except ValueError as e:
                if "Observation spaces do not match" in str(e):
                    logging.warning(f"Model incompatible due to env changes: {e}")
                    logging.info("Starting fresh training.")
                    model = None
                    reset_num_timesteps = True
                    # Clean old logs if incompatible
                    if os.path.exists(tensorboard_log):
                        shutil.rmtree(tensorboard_log)
                else:
                    raise
        else:
            logging.info("Resume requested but no model found. Starting FRESH.")
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
        logging.info(f"Initializing new {args.algo.upper()} model...")

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
                n_steps=4096,  # Increased from 1024 to reduce CPU-GPU bottleneck
                batch_size=16384,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02
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
            "MlpLstmPolicy",
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
    logging.info(f"Training started... Target: {args.total_timesteps} steps")
    logging.info(f"Model: {args.algo.upper()}, Device: {args.device}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=False,
            reset_num_timesteps=reset_num_timesteps
        )
        
        # --- Normal Finish Save ---
        logging.info("Training finished normally.")
        model.save(model_path)
        if hasattr(train_env, 'save'):
            train_env.save(f"{model_path}.pkl")
        logging.info(f"Saved final model to {model_path}")

    except KeyboardInterrupt:
        # --- CTRL+C Save ---
        logging.info("\n\n⚠️ INTERRUPTED! Saving current state before exiting...")
        
        # 1. Save Model
        if model:
            model.save(model_path)
            logging.info(f"✅ Model saved: {model_path}.zip")
        
        # 2. Save Normalization Stats (Critical for Resume)
        if train_env and hasattr(train_env, 'save'):
            train_env.save(f"{model_path}.pkl")
            logging.info(f"✅ Normalization stats saved: {model_path}.pkl")
            
        logging.info("Exiting gracefully.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    # Register the signal handler for debugging hangs
    # signal.signal(signal.SIGINT, debug_signal_handler)
    main()