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
        # 1. FIX: Handle Vectorized Environment (List vs Dict)
        # ---------------------------------------------------------
        infos = self.locals['infos']
        
        # If wrapped in VecEnv, infos is a list of dicts. We want the first one.
        if isinstance(infos, list):
            info = infos[0] 
        else:
            info = infos # Should not happen in SB3 learn(), but safe to handle

        # ---------------------------------------------------------
        # 2. Get Thread 0 Action
        # ---------------------------------------------------------
        # Actions are usually a numpy array [env_0_action, env_1_action, ...]
        actions = self.locals['actions']
        
        # Handle case where actions might be a simple int/float or array
        if isinstance(actions, (np.ndarray, list)):
            action = actions[0]
        else:
            action = actions

        # If action is an array (e.g. Box space), extract the scalar
        if isinstance(action, (np.ndarray, list)):
            action = action[0]

        # ---------------------------------------------------------
        # 3. Extract Data & Store
        # ---------------------------------------------------------
        # Using .get() with defaults prevents crashing if env keys are missing
        current_price = info.get('current_price', info.get('price', 0))
        ema_50 = info.get('ema50', 0)
        portfolio_value = info.get('portfolio_value', 0)

        self.ep_prices.append(current_price)
        self.ep_emas.append(ema_50)
        self.ep_actions.append(action)
        self.ep_portfolio.append(portfolio_value)

        # ---------------------------------------------------------
        # 4. Check for Episode End (Thread 0)
        # ---------------------------------------------------------
        dones = self.locals['dones']
        # Again, handle list vs scalar
        is_done = dones[0] if isinstance(dones, (np.ndarray, list)) else dones

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
        Plots Price/EMA (Top) and Actions (Bottom) to visualize Bull/Bear behavior.
        Sends the image to WandB.
        """
        # 1. Skip if data is insufficient
        if len(self.ep_prices) < 10:
            return

        # 2. Setup Data
        steps = range(len(self.ep_prices))
        prices = np.array(self.ep_prices)
        emas = np.array(self.ep_emas)
        actions = np.array(self.ep_actions)

        # 3. Create Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # --- Top Chart: Market Context (Price + EMA) ---
        ax1.plot(steps, prices, label='Price', color='black', linewidth=1.2)
        
        # Check if EMA data is valid (not all zeros)
        has_ema = any(x > 0 for x in emas)
        if has_ema:
            ax1.plot(steps, emas, label='EMA 50', color='orange', linestyle='--', linewidth=1)
            
            # Highlight Background: Green (Bull) vs Red (Bear)
            # Fill green where Price > EMA
            ax1.fill_between(steps, prices, emas, where=(prices > emas), 
                             color='green', alpha=0.1, interpolate=True, label='Bull Regime')
            # Fill red where Price <= EMA
            ax1.fill_between(steps, prices, emas, where=(prices <= emas), 
                             color='red', alpha=0.1, interpolate=True, label='Bear Regime')

        last_pv = self.ep_portfolio[-1] if self.ep_portfolio else 0
        ax1.set_title(f"Thread 0 Analysis | End Portfolio: {last_pv:.2f}")
        ax1.set_ylabel("Price")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Bottom Chart: Agent Actions ---
        # Colors: Green for Buy (>0), Red for Sell (<0)
        colors = ['green' if a > 0 else 'red' for a in actions]
        
        ax2.bar(steps, actions, color=colors, width=1.0, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Action\n(Sell < 0 < Buy)")
        ax2.set_ylim(-1.1, 1.1) # Assuming action space is -1 to 1
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Steps")

        plt.tight_layout()

        # 4. Log to WandB
        # We wrap in try/except in case WandB is not initialized
        try:
            wandb.log({"trade_analysis/thread_0_regime": wandb.Image(fig)})
        except Exception as e:
            # If wandb isn't active, we just pass to prevent crashing training
            pass

        # 5. Close figure to free memory
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