import argparse
import pandas as pd
import os
import numpy as np
import warnings
import torch
import glob
import multiprocessing
import shutil

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor 

import wandb
from wandb.integration.sb3 import WandbCallback

from trading_env import TradingEnv 

class FeatureImportanceCallback(BaseCallback):
    """
    Analyzes the weights of the Actor network to determine which features 
    are driving decisions. Logs a bar chart to WandB.
    """
    def __init__(self, vp_days, verbose=0):
        super().__init__(verbose)
        self.vp_days = vp_days
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self):
        # 1. Standard Features (6)
        names = ['close_pct', 'vol_norm', 'rsi', 'stoch', 'macd', 'macd_sig']
        # 2. Account (2)
        names.extend(['balance', 'shares'])
        # 3. VP Features
        for days in self.vp_days:
            prefix = f"vp{days}d"
            names.extend([f"{prefix}_poc", f"{prefix}_vah", f"{prefix}_val"])
            # We group the 100 heatmap bins into one label to keep chart readable
            for i in range(100):
                names.append(f"{prefix}_map_{i}")
        return names

    def _on_step(self) -> bool:
        # Log every 5000 steps (calculation is fast, but we don't need spam)
        if self.num_timesteps % 5000 == 0 and wandb.run is not None:
            try:
                # Access Actor's first layer weights
                # Shape: [Hidden_Size, Input_Size]
                # We take absolute value and mean across hidden neurons
                first_layer = self.model.actor.latent_pi[0]
                importance = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
                
                # Aggregate Heatmap bins (otherwise chart is too crowded)
                # We assume the feature order matches _next_observation in trading_env
                
                agg_importance = {}
                heatmap_accumulators = {}
                
                for name, score in zip(self.feature_names, importance):
                    if "_map_" in name:
                        # Aggregate VP heatmaps by day (e.g. vp7d_map)
                        group_name = name.split("_map_")[0] + "_heatmap_avg"
                        heatmap_accumulators[group_name] = heatmap_accumulators.get(group_name, []) + [score]
                    else:
                        agg_importance[name] = score
                
                # Average the heatmap scores
                for key, val_list in heatmap_accumulators.items():
                    agg_importance[key] = np.mean(val_list)

                # Create WandB Data
                data = [[k, v] for k, v in agg_importance.items()]
                table = wandb.Table(data=data, columns=["feature", "importance"])
                
                wandb.log({
                    "analysis/feature_importance": wandb.plot.bar(
                        table, "feature", "importance", 
                        title=f"What is the Bot looking at? (Step {self.num_timesteps})"
                    )
                })
            except Exception as e:
                print(f"Feature importance failed: {e}")
                
        return True


# --- REAL-TIME CALLBACK ---
class RealTimeWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RealTimeWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if wandb.run is None: return True
        
        # In Parallel Envs, 'infos' is a list of N dicts (one per CPU)
        infos = self.locals["infos"][0]
        
        if self.num_timesteps % 1000 == 0:
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
                "realtime/buy_threshold": infos.get("buy_threshold", 0),
                "realtime/sell_threshold": infos.get("sell_threshold", 0),
                "realtime/alpha_entropy": current_alpha,
                "realtime/learning_rate": current_lr,
                
                "realtime/market_context": {
                    "realtime/price_main": infos.get("price", 0),
                    "realtime/price_poc": infos.get("poc", 0), 
                    "realtime/price_vah": infos.get("vah", 0), 
                    "realtime/price_val": infos.get("val", 0), 
                },
                
                "realtime/momentum": {
                    "rsi": infos.get("rsi", 50),
                    "stoch_rsi": infos.get("stoch_rsi", 0.5)
                },
                
                "realtime/macd": {
                    "macd_line": infos.get("macd", 0),
                    "signal_line": infos.get("macd_sig", 0)
                },

                "global_step": self.num_timesteps
            })
            
        if self.num_timesteps % 10000 == 0 and "vp_heatmap" in infos:
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

# --- EVALUATION LISTENER ---
class WandbEvalListener(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.parent is not None:
            mean_reward = self.parent.last_mean_reward
            mean_len = self.parent.last_mean_ep_length
            
            # --- NEW: EXTRACT FINANCIAL METRICS ---
            try:
                # We need to dig through the wrappers to find the real TradingEnv
                # EvalEnv is typically a DummyVecEnv -> Monitor -> TradingEnv
                eval_env = self.parent.eval_env
                # Get the first environment (since we run 1 eval episode usually)
                actual_env = eval_env.envs[0].unwrapped
                
                last_portfolio = getattr(actual_env, 'last_run_net_worth', 0)
                last_roi = getattr(actual_env, 'last_run_roi', 0)
                
                print(f"📈 Eval Finished. ROI: {last_roi:.2f}% | End Port: ${last_portfolio:.0f}")
                
                if mean_reward != -np.inf:
                    wandb.log({
                        "eval/mean_reward": mean_reward,
                        "eval/mean_ep_length": mean_len,
                        "eval/portfolio_value": last_portfolio, # <--- Real $$$
                        "eval/roi_pct": last_roi,               # <--- Real % Gain
                        "global_step": self.num_timesteps
                    })
            except Exception as e:
                print(f"⚠️ Could not extract custom eval metrics: {e}")
                # Fallback to basic logging
                if mean_reward != -np.inf:
                    wandb.log({
                        "eval/mean_reward": mean_reward,
                        "eval/mean_ep_length": mean_len,
                        "global_step": self.num_timesteps
                    })
                    
        return True
    
class EvalDataCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.buy_thresholds = []
        self.sell_thresholds = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        self.buy_thresholds.append(infos.get("buy_threshold", 0))
        self.sell_thresholds.append(infos.get("sell_threshold", 0))
        return True

class CallbackWrapper(gym.Wrapper):
    def __init__(self, env, callback):
        super().__init__(env)
        self.callback = callback

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.callback.locals = {"infos": [info]}
        self.callback._on_step()
        return obs, reward, terminated, truncated, info


# --- CURRICULUM MANAGER ---
class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_phase = 1

    def _on_step(self) -> bool:
        new_phase = 1
        if self.num_timesteps > 500000:
            new_phase = 3
        elif self.num_timesteps > 200000:
            new_phase = 2
            
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            print(f"\n🚀 UPGRADING TO PHASE {new_phase} at step {self.num_timesteps}")
            self.training_env.env_method("set_phase", new_phase)
            if wandb.run is not None:
                wandb.log({"train/curriculum_phase": new_phase, "global_step": self.num_timesteps})
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--data-path", type=str, default="BTCUSDT_data.csv")
    parser.add_argument("--vp-days", nargs='+', type=int, default=[7, 30]) 
    parser.add_argument("--algo", type=str, default="sac") 
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--test-split", type=str, default="2024-01-01")
    parser.add_argument("--resume", nargs='?', const='LATEST', default=None)
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for gradient updates")
    
    # --- NEW DEVICE ARGUMENT ---
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'auto', 'cuda', or 'cpu'")
    
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # --- 0. CLEANUP OLD MODELS ---
    models_dir = f"./models/{args.pair}"
    log_dir = f"./{args.algo}_tb/"

    if args.resume is None:
        print("\n🧹 FRESH START DETECTED")
        
        # 1. Delete Old Models
        if os.path.exists(models_dir):
            print(f"   - Deleting old models in {models_dir}...")
            shutil.rmtree(models_dir)
            os.makedirs(models_dir)
        else:
            os.makedirs(models_dir)

        # 2. Delete Old Logs (Tensorboard)
        if os.path.exists(log_dir):
            print(f"   - Deleting old logs in {log_dir}...")
            shutil.rmtree(log_dir)
            # Note: We don't need to os.makedirs(log_dir) here, 
            # Stable-Baselines3 creates it automatically when needed.
            
        print("✅ Cleanup complete. Starting fresh.")
    else:
        print(f"\n🔄 RESUME DETECTED: Keeping existing models and logs.")


    # --- 1. LOAD DATA ---
    csv_file = args.data_path if args.data_path else "BTCUSDT_data.csv"
    if os.path.exists(csv_file):
        print(f"Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        if 'timestamp' in df.columns: df.rename(columns={'timestamp': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"✂️ Splitting data at {args.test_split}...")
        train_df = df[df.index < args.test_split].copy()
        test_df = df[df.index >= args.test_split].copy()
    else:
        print("❌ Error: File not found.")
        return

    # --- 2. CALCULATE CORES ---
    n_cpu = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Speeding up with {n_cpu} Parallel Environments...")

    # --- 3. WARMUP ---
    print("🔥 Warming up cache on Main Process...")
    warmup_env = TradingEnv(train_df, vp_days=args.vp_days)
    warmup_env.close()
    print("✅ Cache ready. Launching Swarm.")

    # --- 4. INIT PARALLEL ENVIRONMENTS ---
    def make_train_env():
        return Monitor(TradingEnv(train_df, vp_days=args.vp_days))

    env = SubprocVecEnv([make_train_env for _ in range(n_cpu)])
    eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(test_df, vp_days=args.vp_days))])

    # --- 5. MODEL SETUP ---
    policy_kwargs = dict(net_arch=[512, 512]) 
    tensorboard_log = f"./{args.algo}_tb/"
    
    model_path_to_load = None
    if args.resume is not None:
        if args.resume == 'LATEST':
            if os.path.exists(models_dir):
                list_of_files = glob.glob(f"{models_dir}/*.zip")
                if list_of_files:
                    model_path_to_load = max(list_of_files, key=os.path.getmtime)
                    print(f"🔄 Auto-detected: {model_path_to_load}")
        else:
            model_path_to_load = args.resume

    # --- Pass the 'device' argument here ---
    if model_path_to_load and os.path.exists(model_path_to_load):
        print(f"📥 Loading model: {model_path_to_load}")
        if args.algo.lower() == 'sac':
            model = SAC.load(model_path_to_load, env=env, tensorboard_log=tensorboard_log, device=args.device)
        else:
            model = PPO.load(model_path_to_load, env=env, tensorboard_log=tensorboard_log, device=args.device)
    else:
        print(f"✨ Creating NEW {args.algo.upper()} Model on {args.device.upper()}")
        if args.algo.lower() == 'sac':
            model = SAC(
                "MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log=tensorboard_log, 
                learning_rate=3e-4, 
                ent_coef='auto',
                device=args.device, # <--- Explicit device
                batch_size=args.batch_size 
            )
        else:
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log, device=args.device, batch_size=args.batch_size)

    # --- 6. CALLBACKS ---
    callbacks = []
    if args.wandb:
        wandb.init(project="ai-trading-bot", config=vars(args), sync_tensorboard=True, monitor_gym=True, save_code=True)
        callbacks.append(RealTimeWandbCallback())
        callbacks.append(WandbCallback(verbose=2))
        callbacks.append(FeatureImportanceCallback(vp_days=args.vp_days))

    save_freq = max(50000 // n_cpu, 1) # 50000 / 14 = 3571 updates

    callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=models_dir, name_prefix=args.algo))
    callbacks.append(CurriculumCallback())

    eval_listener = WandbEvalListener()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{args.pair}_best_eval',
        log_path=tensorboard_log,
        eval_freq=50000, 
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=eval_listener
    )
    callbacks.append(eval_callback)

    # --- 7. TRAIN ---
    print(f"--- STARTING TRAINING ({args.total_timesteps} steps) ---")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=(model_path_to_load is None))
    model.save(f"{args.algo}_{args.pair}_final")
    print("Training complete.")

# usage example:
# python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 5000000 --wandb --device gpu --batch-size 4096
# or with resume if you berak leaarning
# python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 5000000 --wandb --device cpu --resume --batch-size 2048
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()

