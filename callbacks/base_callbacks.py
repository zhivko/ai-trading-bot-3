import os
import numpy as np
import wandb # Import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

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
    Includes 'Action vs Market Regime' charting for Thread 0.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Buffers for plotting (Thread 0 only)
        self.ep_prices = []
        self.ep_emas = []
        self.ep_actions = []
        self.ep_portfolio = []

    def _on_step(self) -> bool:
        # ---------------------------------------------------------
        # 1. ROBUST FIX: Handle Vectorized Environment & Tuples
        # ---------------------------------------------------------
        infos = self.locals['infos']
        
        # Step A: Unwrap VecEnv list (Batch of infos)
        if isinstance(infos, list):
            # We only care about the first environment (Thread 0) for plotting
            if len(infos) > 0:
                raw_info = infos[0] 
            else:
                raw_info = {} 
        else:
            raw_info = infos

        # Step B: Unwrap Tuple (Gymnasium/SB3 mismatch fix)
        # If the env returns (truncated, info) or similar tuple, find the dict.
        final_info = {}
        if isinstance(raw_info, tuple):
            for item in raw_info:
                if isinstance(item, dict):
                    final_info = item
                    break
        elif isinstance(raw_info, dict):
            final_info = raw_info
        
        # ---------------------------------------------------------
        # 2. Get Thread 0 Action
        # ---------------------------------------------------------
        actions = self.locals['actions']
        
        # Handle cases where action is scalar, list, or numpy array
        if isinstance(actions, (list, np.ndarray)):
            if len(actions) > 0:
                action = actions[0]
            else:
                action = 0
        else:
            action = actions

        # If the specific action is still an array (e.g. continuous space), extract scalar
        if isinstance(action, (list, np.ndarray)):
            try:
                action = action[0]
            except IndexError:
                action = 0

        # ---------------------------------------------------------
        # 3. Extract Data & Store
        # ---------------------------------------------------------
        # Now 'final_info' is guaranteed to be a dict, so .get() is safe.
        current_price = final_info.get('current_price', final_info.get('price', 0))
        ema_50 = final_info.get('ema50', 0)
        portfolio_value = final_info.get('portfolio_value', 0)

        self.ep_prices.append(current_price)
        self.ep_emas.append(ema_50)
        self.ep_actions.append(action)
        self.ep_portfolio.append(portfolio_value)

        # ---------------------------------------------------------
        # 4. Check for Episode End (Thread 0)
        # ---------------------------------------------------------
        dones = self.locals['dones']
        # Handle list vs scalar for 'dones'
        is_done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

        if is_done:
            self._plot_regime_chart()
            # Reset buffers for the next episode
            self.ep_prices = []
            self.ep_emas = []
            self.ep_actions = []
            self.ep_portfolio = []

        return True

    def _plot_regime_chart(self):
        """
        Plots Price/EMA (Top) and Actions (Bottom) to visualize 
        Bull/Bear behavior.
        """
        # Skip empty or short episodes
        if len(self.ep_prices) < 10:
            return

        steps = range(len(self.ep_prices))
        
        # Create Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # --- Top Chart: Market Regime ---
        ax1.plot(steps, self.ep_prices, label='Price', color='black', linewidth=1.2)
        
        # Only plot EMA if valid
        if any(x > 0 for x in self.ep_emas):
            ax1.plot(steps, self.ep_emas, label='EMA 50', color='orange', linestyle='--', linewidth=1)
            
            # Highlight Areas: Green (Price > EMA), Red (Price < EMA)
            prices = np.array(self.ep_prices)
            emas = np.array(self.ep_emas)
            
            # Safe filling (ensure lengths match)
            min_len = min(len(prices), len(emas))
            ax1.fill_between(steps[:min_len], prices[:min_len], emas[:min_len], 
                             where=(prices[:min_len] > emas[:min_len]), color='green', alpha=0.1, label='Bull Zone')
            ax1.fill_between(steps[:min_len], prices[:min_len], emas[:min_len], 
                             where=(prices[:min_len] <= emas[:min_len]), color='red', alpha=0.1, label='Bear Zone')

        last_pv = self.ep_portfolio[-1] if self.ep_portfolio else 0
        ax1.set_title(f"Episode Analysis (Thread 0) | End Portfolio: {last_pv:.2f}")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Bottom Chart: Actions ---
        actions = np.array(self.ep_actions)
        colors = ['green' if a > 0 else 'red' for a in actions]
        
        ax2.bar(steps, actions, color=colors, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Action\n(-1 Sell / +1 Buy)")
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Steps")

        plt.tight_layout()

        # Log to WandB
        try:
            if wandb.run is not None:
                wandb.log({"trade_analysis/thread_0_chart": wandb.Image(fig)})
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
        return True

    def _log_best_to_wandb(self, mean_reward, std_reward):
        if wandb.run is not None:
            wandb.log({
                "best_eval/mean_reward": mean_reward,
                "best_eval/std_reward": std_reward,
                "best_eval/mean_portfolio": self.best_mean_portfolio,
                "best_eval/std_portfolio": self.best_std_portfolio
            }, step=self.num_timesteps)