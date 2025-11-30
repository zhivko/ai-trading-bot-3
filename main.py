import argparse
import pandas as pd
import os
import numpy as np
import warnings
import torch
import glob # <--- Needed to find files
from torch.distributions import Normal
import matplotlib.pyplot as plt

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback
import plotly.express as px

from trading_env import TradingEnv

# --- CUSTOM CHECKPOINT CALLBACK WITH LOGGING ---
class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}")
            self.model.save(path)
            print(f"Checkpoint saved: {path}.zip")
        return True

class WandbEvalListener(BaseCallback):
    """
    This callback runs AFTER the EvalCallback finishes testing.
    It grabs the results from the PARENT callback and logs them.
    """
    def __init__(self, eval_monitor=None, verbose=0):
        super().__init__(verbose)
        self.eval_portfolio_values = []
        self.eval_env = None
        self.eval_monitor = eval_monitor

    def __call__(self, locals_, globals_):
        """Make the callback callable for EvalCallback"""
        self._on_step()

    def _on_step(self) -> bool:
        # The parent is the EvalCallback
        if self.parent is not None:
            # Store reference to eval environment on first call
            if self.eval_env is None:
                self.eval_env = getattr(self.parent, 'eval_env', None)
            
            # Grab the metrics directly from the parent class
            mean_reward = self.parent.last_mean_reward
            mean_len = self.parent.last_mean_ep_length
            
            # Try to extract portfolio value from evaluation monitor if available
            final_portfolio_value = 0
            if self.eval_monitor is not None:
                final_portfolio_value = self.eval_monitor.get_final_portfolio_value()
                print(f"DEBUG: Extracted portfolio_value from eval_monitor: {final_portfolio_value}")
            else:
                # Fallback to old method
                if self.eval_env is not None:
                    try:
                        # Access the wrapped environment to get net_worth
                        # The eval_env is a VecEnv, so we need to access the first env
                        if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
                            first_env = self.eval_env.envs[0]
                            # The Monitor wraps the TradingEnv, so we need to access the inner env
                            if hasattr(first_env, 'env') and hasattr(first_env.env, 'net_worth'):
                                final_portfolio_value = first_env.env.net_worth
                                print(f"DEBUG: Extracted net_worth from first_env.env: {final_portfolio_value}")
                            elif hasattr(first_env, 'net_worth'):
                                final_portfolio_value = first_env.net_worth
                                print(f"DEBUG: Extracted net_worth from first_env: {final_portfolio_value}")
                            else:
                                print("DEBUG: No net_worth attribute found in eval env")
                        else:
                            print("DEBUG: No envs in eval_env")
                    except Exception as e:
                        print(f"Warning: Could not extract portfolio value: {e}")
                        final_portfolio_value = 0
                else:
                    print("DEBUG: eval_env is None")
            
            self.eval_portfolio_values.append(final_portfolio_value)
            
            # Calculate metrics for logging
            mean_portfolio_value = np.mean(self.eval_portfolio_values) if self.eval_portfolio_values else 0
            
            print(f"📈 Sending Eval Metrics to WandB: Reward={mean_reward:.2f}, Portfolio=${final_portfolio_value:.2f}")

            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/mean_ep_length": mean_len,
                "eval/portfolio_value": final_portfolio_value,
                "eval/mean_portfolio_value": mean_portfolio_value,
                "eval/last_10_trades": self.eval_env.envs[0].trade_history[-10:],
                "global_step": self.num_timesteps
            })
        return True

# --- CUSTOM EVAL CALLBACK WITH EARLY STOPPING ---
class CustomEvalCallback(EvalCallback):
    """
    Extends EvalCallback to add early stopping based on evaluation portfolio value.
    Stops training if portfolio value doesn't improve for 5 consecutive evaluations.
    """
    def __init__(self, eval_env, callback_after_eval=None, **kwargs):
        super().__init__(eval_env, callback_after_eval=callback_after_eval, **kwargs)
        # Ensure callback_after_eval is set, as EvalCallback may not set it
        self.callback_after_eval = callback_after_eval
        self.best_portfolio = float('-inf')
        self.no_improve_count = 0
        self.last_eval_count = 0

    def _on_step(self) -> bool:
        # Call parent _on_step to perform evaluation
        continue_training = super()._on_step()

        if not continue_training:
            return False

        # Check if a new evaluation was performed
        current_eval_count = len(self.callback_after_eval.eval_portfolio_values)
        if current_eval_count > self.last_eval_count:
            # Get the latest portfolio value
            latest_portfolio = self.callback_after_eval.eval_portfolio_values[-1]

            if latest_portfolio > self.best_portfolio:
                self.best_portfolio = latest_portfolio
                self.no_improve_count = 0
                print(f"New best portfolio value: ${latest_portfolio:.2f}")
            else:
                self.no_improve_count += 1
                print(f"No improvement in portfolio value. Count: {self.no_improve_count}/5")

                if self.no_improve_count >= 5:
                    print("Early stopping triggered: portfolio value hasn't improved for 5 consecutive evaluations.")
                    return False

            self.last_eval_count = current_eval_count

        return True

# --- EVALUATION MONITORING CALLBACK ---
class EvalMonitorCallback(BaseCallback):
    """
    This callback monitors the evaluation environment during EvalCallback execution
    to capture portfolio metrics that aren't available in the standard EvalCallback.
    """
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.current_episode_rewards = []
        self.current_episode_portfolio_values = []
        self.episode_results = []
        
    def _on_step(self) -> bool:
        # This callback runs during each step of evaluation
        if self.locals.get("dones", [False])[0]:  # Episode finished
            # Extract final portfolio value from info
            infos = self.locals.get("infos", [{}])
            if infos and len(infos) > 0:
                final_info = infos[0]
                portfolio_value = final_info.get("portfolio_value", 0)
                final_reward = final_info.get("reward", 0)
                
                self.current_episode_portfolio_values.append(portfolio_value)
                self.episode_results.append({
                    "portfolio_value": portfolio_value,
                    "final_reward": final_reward
                })
                
        return True
        
    def get_final_portfolio_value(self):
        """Get the final portfolio value from the last evaluation episode"""
        if self.episode_results:
            return self.episode_results[-1]["portfolio_value"]
        return 0

# Feature names list
heatmap7 = [f'heatmap7_{i}' for i in range(100)]
heatmap30 = [f'heatmap30_{i}' for i in range(100)]
norms = ['norm_poc7', 'norm_vah7', 'norm_val7', 'norm_poc30', 'norm_vah30', 'norm_val30']
hvns = ['hvn_count7', 'hvn_avg_dist7', 'hvn_nearest7', 'lvn_count7', 'lvn_avg_dist7', 'lvn_nearest7',
        'hvn_count30', 'hvn_avg_dist30', 'hvn_nearest30', 'lvn_count30', 'lvn_avg_dist30', 'lvn_nearest30']
dists = ['dist_hvn7', 'dist_lvn7', 'rel_poc7', 'in_va7', 'dist_hvn30', 'dist_lvn30', 'rel_poc30', 'in_va30', 'vol', 'imbalance']
macds = ['macd_line', 'macd_signal', 'macd_hist', 'rsi', 'stoch_k', 'stoch_d', 'atr']
sessions = [f'session_{i}' for i in range(24)]
FEATURE_NAMES = heatmap7 + heatmap30 + norms + hvns + dists + macds + sessions

# --- REAL-TIME CALLBACK ---
class RealTimeWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RealTimeWandbCallback, self).__init__(verbose)
        self.portfolio_values = []
        self.episode_count = 0
        self.feature_names = FEATURE_NAMES

    def _on_step(self) -> bool:
        if wandb.run is None: return True
        infos = self.locals["infos"][0]

        # Track portfolio values
        self.portfolio_values.append(infos.get("portfolio_value", 0))

        # Check if episode ended
        if self.locals["dones"][0]:
            sortino = compute_sortino_ratio(self.portfolio_values)
            self.episode_count += 1

            if self.episode_count % 10 == 0:
                # Perform gradient computation
                obs = self.locals["obs"][0]
                state = torch.tensor(obs, dtype=torch.float32, requires_grad=True)

                # Forward through actor
                mean_actions, log_std = self.model.policy.actor(state.unsqueeze(0))
                std = log_std.exp()
                dist = Normal(mean_actions, std)
                action = torch.tensor(self.locals["actions"][0], dtype=torch.float32)
                log_prob = dist.log_prob(action).sum()
                loss = log_prob * sortino
                loss.backward()

                grads = state.grad.abs().detach().cpu().numpy()
                # Normalize to 0-1
                if grads.max() > grads.min():
                    grads = (grads - grads.min()) / (grads.max() - grads.min())
                else:
                    grads = np.zeros_like(grads)

                # Create plot
                plt.figure(figsize=(20, 10))
                plt.bar(range(len(grads)), grads)
                plt.xticks(range(len(grads)), self.feature_names, rotation=90, fontsize=8)
                plt.title(f'Feature Importance (Episode {self.episode_count})')
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                wandb.log({"feature_importance": wandb.Image('feature_importance.png')})
                plt.close()

            # Reset portfolio values for next episode
            self.portfolio_values = []

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
                "realtime/current_phase": infos.get("current_phase", 1),

                # --- CHECK THIS BLOCK ---
                "realtime/market_context": {
                    "realtime/price_main": infos.get("price", 0),
                    "realtime/price_poc": infos.get("poc", 0),  # <--- MUST match trading_env keys
                    "realtime/price_vah": infos.get("vah", 0),  # <--- MUST match trading_env keys
                    "realtime/price_val": infos.get("val", 0),  # <--- MUST match trading_env keys
                    "last_10_trades": infos.get("last_10_trades", []),
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

            # Log trades chart
            self._log_trades_chart()

        return True

    def _log_trades_chart(self):
        """Log a scatter plot of all trades over price"""
        if wandb.run is None: return
        trades = self.locals["env"].envs[0].trade_history
        if len(trades) > 0:
            data = [[t['step'], t['price'], t['type']] for t in trades]
            df = pd.DataFrame(data, columns=["step", "price", "type"])
            fig = px.scatter(df, x="step", y="price", color="type", color_discrete_map={"buy": "green", "sell": "red"})
            wandb.log({
                "trades_over_price": fig
            })


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.current_phase = 1

    def _on_step(self) -> bool:
        num_timesteps = self.num_timesteps
        if num_timesteps < 100000:
            new_phase = 1
        elif num_timesteps < 300000:
            new_phase = 2
        else:
            new_phase = 3

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            # Set on training env
            if hasattr(self.model.env, 'envs'):
                for env_wrapper in self.model.env.envs:
                    if hasattr(env_wrapper, 'env'):
                        env_wrapper.env.set_phase(new_phase)
            # Set on eval env if provided
            if self.eval_env and hasattr(self.eval_env, 'envs'):
                for env_wrapper in self.eval_env.envs:
                    if hasattr(env_wrapper, 'env'):
                        env_wrapper.env.set_phase(new_phase)
            # Log to wandb
            if wandb.run is not None:
                wandb.log({"phase": new_phase, "global_step": num_timesteps})
        return True

def compute_sortino_ratio(portfolio_values):
    """
    Compute Sortino ratio from a list of portfolio values.
    Sortino = mean_return / (downside_std + 1e-9)
    """
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
    sortino = mean_return / (downside_std + 1e-9)
    return sortino

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
        print(f"   📙 Testing:  {len(test_df)} rows")
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
    policy_kwargs = dict(net_arch=[256, 256], optimizer_kwargs={})
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
                print(f"⚠️ No .zip files found in {models_dir}, starting new training instead of resuming.")
                model_path_to_load = None
            else:
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
                ent_coef=0.1
            )
        else:
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log)

    print(f"Initial num_timesteps: {model.num_timesteps}")

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
    checkpoint_save_path = f'./models/{args.pair}'
    os.makedirs(checkpoint_save_path, exist_ok=True)
    callbacks.append(CustomCheckpointCallback(save_freq=50000, save_path=checkpoint_save_path, name_prefix=args.algo))

    # Create evaluation monitoring callback for reliable portfolio tracking
    eval_monitor = EvalMonitorCallback(eval_env)

    # Create the listener
    eval_listener = WandbEvalListener(eval_monitor=eval_monitor)

    # Eval logic
    best_model_save_path = f'./models/{args.pair}_best_eval'
    os.makedirs(best_model_save_path, exist_ok=True)
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=f'./sac_tb/',
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        callback_after_eval=eval_listener
    )
    callbacks.append(eval_monitor)
    callbacks.append(eval_callback)

    curriculum_callback = CurriculumCallback(eval_env=eval_env)
    callbacks.append(curriculum_callback)

    # --- 5. TRAIN ---
    print(f"\n--- 3. STARTING TRAINING ({args.total_timesteps} steps) ---")
    
    # reset_num_timesteps=False allows logging to continue where it left off
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=(model_path_to_load is None))
    
    model.save(f"{args.algo}_{args.pair}_final")
    print("Training complete.")


# usage
# python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 1000000 --wandb
# or
# python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 1000000 --wandb --resume
if __name__ == "__main__":
    main()