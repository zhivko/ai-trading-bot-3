import os
import numpy as np
import wandb # Import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('..')
from fetch_metrics import generate_metrics


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard/wandb.
    Now supports DATE/TIME on X-Axis using info['date'].
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Buffers
        self.ep_prices = []
        self.ep_emas = []
        self.ep_actions = []
        self.ep_portfolio = []
        self.ep_dates = [] # <--- Buffer for dates

    def _on_step(self) -> bool:
        # 1. Handle VecEnv & Tuples (Robust extraction)
        infos = self.locals['infos']
        
        # Unwrap VecEnv list
        if isinstance(infos, list):
            raw_info = infos[0] if len(infos) > 0 else {}
        else:
            raw_info = infos

        # Unwrap Tuple (Gymnasium/SB3 mismatch fix)
        final_info = {}
        if isinstance(raw_info, tuple):
            for item in raw_info:
                if isinstance(item, dict):
                    final_info = item
                    break
        elif isinstance(raw_info, dict):
            final_info = raw_info

        # 2. Get Thread 0 Action
        actions = self.locals['actions']
        if isinstance(actions, (list, np.ndarray)):
            action = actions[0] if len(actions) > 0 else 0
        else:
            action = actions
        
        if isinstance(action, (list, np.ndarray)):
            try: action = action[0]
            except IndexError: action = 0

        # 3. Extract Data safely
        current_price = final_info.get('current_price', final_info.get('price', 0))
        ema_50 = final_info.get('ema50', 0)
        portfolio_value = final_info.get('portfolio_value', 0)
        
        # --- GET DATE ---
        # Your env sends 'date', we fallback to step count if missing
        current_date = final_info.get('date', f"{self.n_calls}")

        # 4. Store
        self.ep_prices.append(current_price)
        self.ep_emas.append(ema_50)
        self.ep_actions.append(action)
        self.ep_portfolio.append(portfolio_value)
        self.ep_dates.append(current_date) # <--- Store the date

        # 5. Check Done
        dones = self.locals['dones']
        is_done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

        if is_done:
            self._plot_regime_chart()
            # Reset buffers
            self.ep_prices = []
            self.ep_emas = []
            self.ep_actions = []
            self.ep_portfolio = []
            self.ep_dates = []

        return True

    def _plot_regime_chart(self):
        if len(self.ep_prices) < 10:
            return

        # Prepare Data
        # We use integers for plotting alignment, then swap labels later
        steps = np.arange(len(self.ep_prices)) 
        prices = np.array(self.ep_prices)
        emas = np.array(self.ep_emas)
        actions = np.array(self.ep_actions)
        dates = self.ep_dates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # --- Top Chart (Price) ---
        ax1.plot(steps, prices, label='Price', color='black', linewidth=1.2)
        if any(x > 0 for x in emas):
            ax1.plot(steps, emas, label='EMA 50', color='orange', linestyle='--', linewidth=1)
            
            min_len = min(len(prices), len(emas))
            ax1.fill_between(steps[:min_len], prices[:min_len], emas[:min_len], 
                             where=(prices[:min_len] > emas[:min_len]), color='green', alpha=0.1, label='Bull Zone')
            ax1.fill_between(steps[:min_len], prices[:min_len], emas[:min_len], 
                             where=(prices[:min_len] <= emas[:min_len]), color='red', alpha=0.1, label='Bear Zone')

        last_pv = self.ep_portfolio[-1] if self.ep_portfolio else 0
        ax1.set_title(f"Episode Analysis (Thread 0) | End Portfolio: {last_pv:.2f}")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Bottom Chart (Actions) ---
        colors = ['green' if a > 0 else 'red' for a in actions]
        ax2.bar(steps, actions, color=colors, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Action")
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)

        # --- X-AXIS DATE FORMATTING ---
        # 1. Select ~8 evenly spaced indices to show as ticks
        num_ticks = 8
        tick_indices = np.linspace(0, len(steps) - 1, num_ticks, dtype=int)
        
        # 2. Get the dates corresponding to those indices
        tick_labels = []
        for idx in tick_indices:
            raw_date = dates[idx]
            # Convert timestamp/datetime to string
            if hasattr(raw_date, 'strftime'):
                d_str = raw_date.strftime('%Y-%m-%d\n%H:%M')
            else:
                d_str = str(raw_date)[:16] # Fallback string truncation
            tick_labels.append(d_str)

        # 3. Apply labels
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=9)
        # -----------------------------

        plt.tight_layout()

        try:
            # --- NEW: Calculate Overtrading Metrics ---
            actions = np.array(self.ep_actions)

            # Count how many times the sign changes (Buy -> Sell or Sell -> Buy)
            # We use sign(actions) and look for differences
            action_signs = np.sign(actions)
            trade_count = np.count_nonzero(np.diff(action_signs))

            # Calculate Turnover Rate (Percentage of steps where a trade occurred)
            turnover_rate = trade_count / len(actions)

            if wandb.run is not None:
                wandb.log({
                    "trade_analysis/thread_0_chart": wandb.Image(fig), # Your existing chart
                    "metrics/trades_per_episode": trade_count,         # <--- NEW
                    "metrics/turnover_rate": turnover_rate             # <--- NEW
                })
        except Exception:
            pass

        plt.close(fig)


class CustomEvalCallback(EvalCallback):
    """
    Custom EvalCallback that logs evaluation metrics including portfolio (networth) to WandB.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_on_new_best = None  # Avoid init_callback error by not setting to function
        self.best_mean_portfolio = -np.inf
        self.best_std_portfolio = 0

    def _evaluate_with_portfolio(self):
        portfolio_values = []
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                action = action[0]
                obs, reward, done, info = self.eval_env.step([action])
                reward = reward[0]
                done = done[0]
                info = info[0]
                episode_reward += reward
            episode_rewards.append(episode_reward)

            if isinstance(info, list):
                current_info = info[0]
            else:
                current_info = info

            portfolio_values.append(current_info.get('portfolio_value', episode_reward))

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_portfolio = np.mean(portfolio_values)
        std_portfolio = np.std(portfolio_values)
        return mean_reward, std_reward, mean_portfolio, std_portfolio

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward, mean_portfolio, std_portfolio = self._evaluate_with_portfolio()
            if self.log_path is not None:
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                self.logger.record("eval/mean_portfolio", mean_portfolio)
                self.logger.record("eval/std_portfolio", std_portfolio)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_std_reward = std_reward
                self.best_mean_portfolio = mean_portfolio
                self.best_std_portfolio = std_portfolio
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self._log_best_to_wandb(mean_reward, std_reward)
            # Generate metrics after evaluation
            if wandb.run is not None:
                generate_metrics()
        return True

    def _log_best_to_wandb(self, mean_reward, std_reward):
        if wandb.run is not None:
            wandb.log({
                "best_eval/mean_reward": mean_reward,
                "best_eval/std_reward": std_reward,
                "best_eval/mean_portfolio": self.best_mean_portfolio,
                "best_eval/std_portfolio": self.best_std_portfolio
            }, step=self.num_timesteps)