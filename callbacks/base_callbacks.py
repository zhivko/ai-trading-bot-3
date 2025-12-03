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
    Custom callback for plotting:
    1. Price vs EMA vs Action (Regime Analysis)
    2. Financial Metrics (Sharpe, Drawdown, Benchmark Comparison)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_prices = []
        self.ep_emas = []
        self.ep_actions = []
        self.ep_portfolio = []
        self.ep_dates = [] 

    def _on_step(self) -> bool:
        # 1. Robust Info Extraction
        infos = self.locals['infos']
        info = infos[0] if isinstance(infos, list) else infos
        
        # Handle tuple unpacking if necessary
        if isinstance(info, tuple):
            for item in info:
                if isinstance(item, dict):
                    info = item
                    break
            if not isinstance(info, dict): info = {}

        # 2. Get Data
        actions = self.locals['actions']
        action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions
        if isinstance(action, (list, np.ndarray)):
            try: action = action[0]
            except IndexError: action = 0

        current_price = info.get('current_price', info.get('price', 0))
        ema_50 = info.get('ema50', 0)
        portfolio_value = info.get('portfolio_value', 0)
        current_date = info.get('timestamp', f"{self.n_calls}")

        # 3. Store
        self.ep_prices.append(current_price)
        self.ep_emas.append(ema_50)
        self.ep_actions.append(action)
        self.ep_portfolio.append(portfolio_value)
        self.ep_dates.append(current_date)

        # 4. Episode End Logic
        dones = self.locals['dones']
        is_done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

        if is_done:
            self._calculate_financial_metrics() # <--- NEW: Calculate Stats
            self._plot_regime_chart()
            
            # Reset
            self.ep_prices = []
            self.ep_emas = []
            self.ep_actions = []
            self.ep_portfolio = []
            self.ep_dates = []

        return True

    def _calculate_financial_metrics(self):
        """Calculates Sharpe, Sortino, Drawdown, Calmar, and Benchmark comparison."""
        if len(self.ep_portfolio) < 2:
            return

        # Convert lists to arrays
        portfolio = np.array(self.ep_portfolio)
        prices = np.array(self.ep_prices)

        # 1. Calculate Returns
        # pct_change = (now - prev) / prev
        returns = np.diff(portfolio) / portfolio[:-1]

        # 2. Sharpe Ratio (Annualized assuming hourly data)
        # periods_per_year = 365 * 24 = 8760
        std_dev = np.std(returns)
        if std_dev > 0:
            sharpe = (np.mean(returns) / std_dev) * np.sqrt(8760)
        else:
            sharpe = 0

        # 2.5 Sortino Ratio (Annualized assuming hourly data)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = (np.mean(returns) / downside_std) * np.sqrt(8760) if downside_std > 0 else 0
        else:
            sortino = 0

        # 3. Max Drawdown
        # Peak so far
        running_max = np.maximum.accumulate(portfolio)
        # Drawdown = (current - peak) / peak
        drawdown = (portfolio - running_max) / running_max
        max_drawdown = np.min(drawdown) # This will be negative, e.g., -0.20 for 20% drop

        # 4. Benchmark (Buy & Hold) Return
        if prices[0] > 0:
            bnh_return = (prices[-1] - prices[0]) / prices[0]
            strategy_return = (portfolio[-1] - portfolio[0]) / portfolio[0]
        else:
            bnh_return = 0
            strategy_return = 0

        # 4.5 Calmar Ratio
        if len(returns) > 0:
            periods_per_year = 8760
            annualized_return = (1 + strategy_return) ** (periods_per_year / len(returns)) - 1
            calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        else:
            annualized_return = 0
            calmar = 0

        # 5. Log to WandB
        try:
            if wandb.run is not None:
                wandb.log({
                    "financial/sharpe_ratio": sharpe,
                    "financial/sortino_ratio": sortino,
                    "financial/max_drawdown": max_drawdown,
                    "financial/calmar_ratio": calmar,
                    "financial/annualized_return": annualized_return,
                    "financial/strategy_return": strategy_return,
                    "financial/benchmark_return": bnh_return,
                    "financial/outperformance": strategy_return - bnh_return,
                    "financial/initial_capital": portfolio[0],
                    "financial/final_networth": portfolio[-1]
                })
        except Exception:
            pass

    def _plot_regime_chart(self):
        # (Keep your existing charting code exactly as it is)
        if len(self.ep_prices) < 10: return
        steps = np.arange(len(self.ep_prices)) 
        prices = np.array(self.ep_prices)
        emas = np.array(self.ep_emas)
        actions = np.array(self.ep_actions)
        dates = self.ep_dates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(steps, prices, label='Price', color='black', linewidth=1.2)
        if any(x > 0 for x in emas):
            ax1.plot(steps, emas, label='EMA 50', color='orange', linestyle='--', linewidth=1)
            min_len = min(len(prices), len(emas))
            ax1.fill_between(steps[:min_len], prices[:min_len], emas[:min_len], where=(prices[:min_len] > emas[:min_len]), color='green', alpha=0.1, label='Bull')
            ax1.fill_between(steps[:min_len], prices[:min_len], emas[:min_len], where=(prices[:min_len] <= emas[:min_len]), color='red', alpha=0.1, label='Bear')
        
        last_pv = self.ep_portfolio[-1] if self.ep_portfolio else 0
        ax1.set_title(f"Thread 0 | PV: {last_pv:.2f}")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        colors = ['green' if a > 0 else 'red' for a in actions]
        ax2.bar(steps, actions, color=colors, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Action")
        
        # Date labels
        num_ticks = 8
        tick_indices = np.linspace(0, len(steps) - 1, num_ticks, dtype=int)
        tick_labels = []
        for idx in tick_indices:
            raw_date = dates[idx]
            if hasattr(raw_date, 'strftime'): d_str = raw_date.strftime('%Y-%m-%d\n%H:%M')
            else: d_str = str(raw_date)[:16]
            tick_labels.append(d_str)
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=8)

        plt.tight_layout()
        try:
            if wandb.run is not None: wandb.log({"trade_analysis/thread_0_chart": wandb.Image(fig)})
        except Exception: pass
        plt.close(fig)


class CustomEvalCallback(EvalCallback):
    """
    Custom EvalCallback that logs evaluation metrics including portfolio (networth) to WandB.
    """
    def __init__(self, *args, **kwargs):
        self.test_split = kwargs.pop('test_split', None)
        self.pair = kwargs.pop('pair', None)
        self.initial_balance = kwargs.pop('initial_balance', None)
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
                    metadata = {}
                    if self.test_split: metadata["test_split"] = self.test_split
                    if self.pair: metadata["pair"] = self.pair
                    if self.initial_balance: metadata["initial_balance"] = self.initial_balance
                    metadata["end_networth"] = self.best_mean_portfolio
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'), metadata=metadata)
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