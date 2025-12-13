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
import warnings
logger = logging.getLogger(__name__)

# Suppress stable_baselines3 deprecation warnings
warnings.filterwarnings("ignore", message="get_schedule_fn.*deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="constant_fn.*deprecated", category=UserWarning)

# Matplotlib backend fix for server environments
import matplotlib
matplotlib.use('Agg')

# RL & Gym
import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import SAC, PPO, A2C, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed, ConstantSchedule
from stable_baselines3.common.env_checker import check_env

# WandB
import wandb
from wandb.integration.sb3 import WandbCallback
wandb.require("core")

# Custom Modules
from enhanced_trading_env import EnhancedTradingEnv
from phase_manager import PhaseManager
from callbacks.base_callbacks import TensorboardCallback, CustomEvalCallback, ProgressBarCallback
from callbacks.recurrent_saliency import RecurrentFeatureSaliencyCallback
from volume_profile import get_rolling_vp

# --- Debug Signal Handler for Thread Stack Traces ---
def debug_signal_handler(signum, frame):
    """
    Catches Ctrl+C (SIGINT) and prints the stack trace of ALL running threads
    before exiting. This helps debug hangs by showing where each thread is stuck.
    """
    logger.info(f"\n\n!!! CAUGHT CTRL+C (Signal {signum}) !!!")
    logger.info("Dumping stack traces for all running threads to see where it hangs...\n")

    # Map thread IDs to their names for cleaner output
    id2name = {t.ident: t.name for t in threading.enumerate()}

    for thread_id, stack in sys._current_frames().items():
        name = id2name.get(thread_id, f"Thread ID {thread_id}")
        logger.info(f"--- Stack trace for Thread: {name} ---")
        stack_trace = ''.join(traceback.format_stack(stack))
        logger.info(stack_trace)
        logger.info("-" * 40 + "\n")

    logger.info("Exiting application...")
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
    parser.add_argument("--buy-threshold", type=float, default=0.0, help="Threshold to trigger buy action")
    parser.add_argument("--sell-threshold", type=float, default=0.0, help="Threshold to trigger sell action")
    
    # Environment Config
    # REVERTED TO YOUR DEFAULT: [7, 30]
    parser.add_argument("--vp-days", type=int, nargs='+', default=[7, 30], help="Volume Profile days (e.g. 7 30)")
    parser.add_argument("--vp-bins", type=int, default=40, help="Volume Profile bins")
    parser.add_argument("--window-size", type=int, default=50, help="Observation window size")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--phase", type=int, default=1, help="Curriculum phase (1=profit, 2=sortino, 3=mdd)")
    parser.add_argument('--total-phases', type=int, default=10, help='Total number of curriculum phases')

    # RL Config
    parser.add_argument("--algo", type=str, default="recurrentppo", choices=["sac", "ppo", "a2c", "td3", "recurrentppo"], help="RL Algorithm")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    
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


def phased_training_loop(model, train_env, phase_manager, total_timesteps, callbacks, algo, pair, reset_num_timesteps=True):
    """
    Implement phased training with entropy coefficient annealing and threshold adjustments.
    """
    steps_per_phase = phase_manager.get_steps_per_phase(total_timesteps)
    total_phases = phase_manager.total_phases
    
    logger.info(f"Starting phased training with {total_phases} phases")
    logger.info(f"Steps per phase: {steps_per_phase:,}")
    logger.info(f"Total target steps: {total_timesteps:,}")
    
    # Calculate checkpoint frequency (every 2 phases)
    checkpoint_freq = steps_per_phase * 2
    
    for phase in range(1, total_phases + 1):
        # Get phase parameters
        phase_params = phase_manager.get_phase_params(phase)
        
        # Log phase start
        phase_manager.log_phase_start(phase, steps_per_phase)
        
        # Update environment thresholds
        train_env.env_method("set_thresholds", phase_params['buy_threshold'], phase_params['sell_threshold'])

        # Update model entropy coefficient
        model.ent_coef = phase_params['entropy_coef']

        logger.info(f"Phase {phase}: Updated entropy_coef to {phase_params['entropy_coef']:.6f}")
        logger.info(f"Phase {phase}: Updated buy_threshold to {phase_params['buy_threshold']:.3f}, sell_threshold to {phase_params['sell_threshold']:.3f}")

        # Raise thresholds in early phases if needed (override phase_manager values)
        # Set up callbacks for this phase
        phase_callbacks = callbacks.copy()
        
        # Add checkpoint callback for this phase (save every 2 phases)
        if phase % 2 == 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=f"./checkpoints/{algo}_{pair}_phase{phase}",
                name_prefix=f"{algo}_phase{phase}"
            )
            phase_callbacks.append(checkpoint_callback)
            logger.info(f"Phase {phase}: Checkpoint callback added (saving every {checkpoint_freq:,} steps)")
        
        # Create callback list for this phase
        phase_callback_list = CallbackList(phase_callbacks)
        
        # Calculate steps for this phase
        if phase == total_phases:
            # Last phase: use remaining steps
            phase_steps = total_timesteps - ((phase - 1) * steps_per_phase)
        else:
            phase_steps = steps_per_phase
        
        logger.info(f"Phase {phase}: Training for {phase_steps:,} steps")

        # Train for this phase
        logger.debug(f"Starting model.learn for phase {phase}")
        try:
            model.learn(
                total_timesteps=phase_steps,
                callback=phase_callback_list,
                progress_bar=False,
                reset_num_timesteps=reset_num_timesteps and phase == 1
            )
        except Exception as e:
            logger.error(f"Error during learning in phase {phase}: {e}", exc_info=True)
            raise
        
        # Save intermediate model after each phase
        phase_model_path = f"./models/{algo}_{pair}_phase{phase}"
        model.save(phase_model_path)
        if hasattr(train_env, 'save'):
            train_env.save(f"{phase_model_path}.pkl")
        logger.info(f"Phase {phase}: Saved model to {phase_model_path}")
        
        # Update for next phase
        reset_num_timesteps = False
        
        logger.info(f"Phase {phase} completed successfully!\n")
    
    logger.info("All phases completed successfully!")
    return model

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
    df = df.bfill().ffill()
    
    return df

# ---------------------------------------------------------
# 3. Main Execution Flow
# ---------------------------------------------------------
def main():
    # Delete old log file to start fresh
    if os.path.exists('logs/ml.log'):
        os.remove('logs/ml.log')

    env_log_files = glob.glob('logs/env_*.log')
    for f in env_log_files:
        os.remove(f)

    # Delete old callback log files
    callback_log_files = glob.glob('logs/*callback*_*.log')
    for f in callback_log_files:
        os.remove(f)

    # Use local time in loggers
    logging.Formatter.converter = time.localtime

    logging.basicConfig(filename='logs/ml.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # Add console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    # Filter to only log from our own files
    class OurFilter(logging.Filter):
        def __init__(self, allowed_files):
            super().__init__()
            self.allowed_files = allowed_files

        def filter(self, record):
            return record.filename in self.allowed_files

    # List of our source files
    our_files = [
        'main.py',
        'enhanced_trading_env.py',
        'volume_profile.py',
        'data_fetcher.py',
        'backtest.py',
        'test_trading_env.py',
        'temp_check.py',
        'visualize_predictions.py',
        'fetch_metrics.py'
    ]
    # Add callback files
    callback_files = glob.glob('callbacks/*.py')
    our_files.extend([os.path.basename(f) for f in callback_files])

    # Apply filter to the root logger's handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(OurFilter(our_files))

    logger.info("Starting main function...")
    args = parse_args()
    logger.info(f"Parsed args: {args}")
    set_random_seed(args.seed)
    logger.info("Random seed set.")
    
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
        'buy_threshold': 0.0,        # ← Allow any signal
        'sell_threshold': 0.0,
        'trading_fee_multiplier': args.trading_fee,
        'phase': args.phase,
        'total_phases': args.total_phases,
        'min_trade_value_usd': 5.0,  # ← Critical: increased to reduce dust noise
        'pair': args.pair,
        'timeframe': args.timeframe,
        'split_date': args.test_split,
    }
    logging.info(f"Env kwargs: {env_kwargs}")

    # Validate environment API before parallelization
    raw_env = EnhancedTradingEnv(
        df=train_df,
        precalculated_vp=vp_data_train,
        **env_kwargs
    )

    check_env(raw_env, warn=True)
    logger.info("Environment validation passed successfully!")
    raw_env.close()

    logging.info(f"Creating {args.n_envs} parallel environments...")

    # Training Env
    logging.info("Creating training environment...")
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs['df'] = train_df
    train_env_kwargs['precalculated_vp'] = vp_data_train

    train_env = make_vec_env(
        EnhancedTradingEnv,
        # Temporarily use DummyVecEnv with 1 env for debugging (avoids EOFError by running in main process)
        n_envs=10,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,              #DummyVecEnv,
        env_kwargs=train_env_kwargs
    )
    logger.info("Training environment created.")

    # FIX: Do not apply VecNormalize here if we are going to load it later in Resume block
    # This prevents "Double Normalization" (z-score of a z-score)
    if not args.resume:
        logger.info("Applying VecNormalize to training env...")
        # CHANGE: norm_obs=False. We scale manually in the Env now.
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_obs=10., clip_reward=10.)
        logger.info("VecNormalize applied.")
        # If needed, disable VecNormalize temporarily by commenting the above and using raw train_env
        logger.info(f"Train env type after normalization: {type(train_env)}")

    # No VecFrameStack needed for RecurrentPPO - LSTM handles temporal dependencies internally

    # Evaluation Env (Raw for accurate metrics)
    logger.info("Creating evaluation environment...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['df'] = test_df
    eval_env_kwargs['precalculated_vp'] = vp_data_test

    eval_env = DummyVecEnv([lambda: EnhancedTradingEnv(**eval_env_kwargs)])
    logger.info("Evaluation environment created.")
    logger.info(f"Eval env type before normalization: {type(eval_env)}")

    # No VecFrameStack needed for RecurrentPPO - LSTM handles temporal dependencies internally

    # FIX: Eval env MUST be normalized using Training stats, otherwise agent sees garbage
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.)
    eval_env.training = False  # Do not update stats during evaluation
    # REMOVED: We do not need to sync obs_rms because we are using norm_obs=False
    # and doing manual scaling inside the environment.
    # If re-enabling norm_obs=True, add: eval_env.obs_rms = train_env.obs_rms

    # Create dummy env for saliency callback (skip for RecurrentPPO due to LSTM compatibility issues)
    saliency_callback = None
    # if args.algo.lower() != 'recurrentppo':
    #     dummy_env = eval_env.envs[0]
    #     saliency_callback = FeatureSaliencyCallback(dummy_env=dummy_env, check_freq=10000)

    # --- W&B Setup ---
    logger.info("Setting up W&B..." if args.wandb else "Skipping W&B setup.")
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
            logger.info(f"🔄 Resuming W&B Run ID: {run_id}")
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
        logger.info(f"W&B initialized (Run: {run_name})")

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f"./checkpoints/{args.algo}_{args.pair}",
        name_prefix=args.algo
    )
    
    tensorboard_callback = TensorboardCallback(verbose=1, buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)

    progress_callback = ProgressBarCallback(update_interval=1000)
    callbacks = [progress_callback, tensorboard_callback, checkpoint_callback]
    #callbacks = [tensorboard_callback, checkpoint_callback]

    #if saliency_callback is not None:
    #    callbacks.append(saliency_callback)

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

    # Note: Phase switching is now handled in the phased_training_loop
    # No need for old PhaseSwitchCallback in the callback list

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
            logger.info(f"✅ Successfully retrieved {len(feature_names)} feature names from environment.")
            logger.info(f"Feature names: {feature_names}")
        except Exception as e:
            logger.warning(f"⚠️ env_method failed to get names ({e}). Trying attribute access...")
            try:
                # Method B: Try accessing the attribute (Backup)
                feature_names = train_env.get_attr("feature_names", indices=0)[0]
                logger.info(f"✅ Retrieved {len(feature_names)} feature names via attribute.")
            except Exception as e2:
                # Method C: Fallback to generics
                obs_dim = train_env.observation_space.shape[0]
                logger.warning(f"❌ Could not retrieve feature names ({e2}). Generating {obs_dim} generic labels.")
                feature_names = [f"F_{i}" for i in range(obs_dim)]

        # --- RE-ENABLE SALIENCY ---
        #saliency_callback = RecurrentFeatureSaliencyCallback(
        #    check_freq=eval_freq_adjusted,  # Run every 50k steps (Heavy computation)
        #    save_path=os.path.join(log_dir, "saliency"),
        #    feature_names=feature_names,
        #    verbose=1
        #)
        #callbacks.append(saliency_callback)
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
    best_model_path = f"{log_dir}/best_model_{args.algo}.zip"
    norm_path = f"{log_dir}/vec_normalize_{args.algo}.pkl"
    
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
                    # === FIX STARTS HERE ===
                    # 1. Load using the INNER environment (unwrap the empty wrapper we just made)
                    temp_env = VecNormalize.load(f"{load_path}.pkl", train_env.venv)

                    # 2. Check if observation dimensions match
                    if temp_env.obs_rms.mean.shape != train_env.observation_space.shape:
                        raise ValueError("Observation spaces do not match (pkl vs env)")

                    # 3. Replace the training env
                    train_env = temp_env
                    train_env.training = True
                    
                    # 4. REMOVED: We do not need to sync obs_rms because we are using norm_obs=False
                    # and doing manual scaling inside the environment.
                    # # (Because train_env is now a different object)
                    # eval_env.obs_rms = train_env.obs_rms
                    # === FIX ENDS HERE ===
                # CRITICAL: Do not reset steps when resuming
                reset_num_timesteps = False
                logger.info(f"   > Resuming from Global Step: {model.num_timesteps}")
            except (ValueError, EOFError) as e:  # Catch more errors
                logger.warning(f"Model incompatible due to env changes: {e}")
                logger.info("Starting fresh training.")
                model = None
                reset_num_timesteps = True
                # Clean old logs if incompatible
                if os.path.exists(tensorboard_log):
                    shutil.rmtree(tensorboard_log)
        else:
            logger.info("Resume requested but no model found. Starting FRESH.")
            # Clean logs if we failed to find a model to resume
            if os.path.exists(tensorboard_log):
                shutil.rmtree(tensorboard_log)
    else:
        logger.info("Starting FRESH training. Cleaning up old models...")
        
        # 1. Delete Model ZIPs
        model_pattern = f"./models/{args.algo}_*.zip"
        model_files = glob.glob(model_pattern)
        for f in model_files:
            try:
                os.remove(f)
                logger.info(f"Deleted old model: {f}")
            except OSError as e:
                logger.error(f"Error deleting {f}: {e}")

        # 2. Delete VecNormalize PKLs (Aggressive Pattern)
        # matches "recurrentppo_BTCUSDT.pkl" and "vec_normalize_recurrentppo.pkl"
        pkl_files = glob.glob(f"./models/*.pkl") + glob.glob(f"./logs/*.pkl")
        for f in pkl_files:
            if args.algo in f: # Only delete files for this algorithm
                try:
                    os.remove(f)
                    logger.info(f"Deleted old normalization stats: {f}")
                except OSError as e:
                    logger.error(f"Error deleting {f}: {e}")

        # ... (rest of cleanup)

        # delete checkpoints for algo
        chk_pattern = f"./checkpoints/{args.algo}_*/**/*.zip"
        chk_files = glob.glob(chk_pattern, recursive=True)
        for f in chk_files:
            os.remove(f)

        # delete tensorboard logs for algo
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)

     

    if model is None:
        logger.info(f"Initializing new {args.algo.upper()} model...")

        # Clean old logs only if starting fresh
        if os.path.exists(tensorboard_log):
            shutil.rmtree(tensorboard_log)

        # Clean old checkpoints
        chk_dir = f"checkpoints/{args.algo}_{args.pair}"
        if os.path.exists(chk_dir):
            shutil.rmtree(chk_dir)

        # Set model hyperparameters
        model_kwargs = {
            "verbose": 1,
            "ent_coef": 0.01,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 128,
            "clip_range": 0.25,  # Slightly larger — allows faster policy adaptation to reversals
            "gae_lambda": 0.95,
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Initialize Phase Manager ---
    phase_manager = PhaseManager(
        total_phases=args.total_phases,
        initial_entropy=0.15,    # Increased from 0.08 — more early exploration to discover switching benefits
        final_entropy=0.0001,    # End low for exploitation
        initial_buy_threshold=0.15,  # Start with tighter buy thresholds
        final_buy_threshold=0.35,    # End with more selective buy thresholds
        initial_sell_threshold=-0.15, # Start with tighter sell thresholds
        final_sell_threshold=-0.35    # End with more selective sell thresholds
        # For now, assume defaults; if log shows 0.000000 in Phase 1, check PhaseManager.get_phase_params(1)
        # and adjust initial_entropy accordingly in PhaseManager __init__
    )

    # Pass phase_manager to environment kwargs
    env_kwargs['phase_manager'] = phase_manager
    
    # Set initial environment thresholds
    initial_params = phase_manager.get_phase_params(1)
    train_env.env_method("set_thresholds", initial_params['buy_threshold'], initial_params['sell_threshold'])
    logger.info(f"Set initial buy_threshold to {initial_params['buy_threshold']:.3f}, sell_threshold to {initial_params['sell_threshold']:.3f}")

    # --- Initialize Model ---
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"./logs/{args.algo}_tensorboard",
        device=args.device,
        **model_kwargs  # Use the dictionary defined above to avoid inconsistencies
    )

    reset_num_timesteps = True

    # --- Phased Training ---
    logger.info(f"Starting phased training... Target: {args.total_timesteps} steps")
    logger.info(f"Model: {args.algo.upper()}, Device: {args.device}")
    logger.info(f"Total phases: {args.total_phases}")

    try:
        # Use phased training instead of single training loop
        model = phased_training_loop(
            model=model,
            train_env=train_env,
            phase_manager=phase_manager,
            total_timesteps=args.total_timesteps,
            callbacks=callbacks,
            algo=args.algo,
            pair=args.pair,
            reset_num_timesteps=reset_num_timesteps
        )
        
        # --- Normal Finish Save ---
        logger.info("Training finished normally.")
        model.save(model_path)
        if hasattr(train_env, 'save'):
            train_env.save(f"{model_path}.pkl")
        logger.info(f"Saved final model to {model_path}")

        # Finish WandB run
        if args.wandb:
            wandb.finish()
            logger.info("WandB run finished.")

    except KeyboardInterrupt:
        # --- CTRL+C Save ---
        logger.info("\n\n⚠️ INTERRUPTED! Saving current state before exiting...")
        
        # 1. Save Model
        if model:
            model.save(model_path)
            logger.info(f"✅ Model saved: {model_path}.zip")
        
        # 2. Save Normalization Stats (Critical for Resume)
        if train_env and hasattr(train_env, 'save'):
            train_env.save(f"{model_path}.pkl")
            logger.info(f"✅ Normalization stats saved: {model_path}.pkl")

        # Finish WandB run
        if args.wandb:
            wandb.finish()
            logger.info("WandB run finished.")

        logger.info("Exiting gracefully.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    # Register the signal handler for debugging hangs
    # signal.signal(signal.SIGINT, debug_signal_handler)
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    main()