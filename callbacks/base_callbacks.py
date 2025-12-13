import os
import numpy as np
import wandb # Import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import matplotlib
import sys
from fetch_metrics import generate_metrics
import logging
import threading
from tqdm import tqdm

matplotlib.use('Agg')
sys.path.append('..')

class ProgressBarCallback(BaseCallback):
    """
    A custom progress bar that updates less frequently to prevent console spam.
    """
    def __init__(self, update_interval=5000):
        super().__init__()
        self.pbar = None
        self.update_interval = update_interval
        self.last_update_step = 0
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            thread_name = threading.current_thread().name
            handler = logging.FileHandler(f"{self.__class__.__name__.lower()}_{thread_name}.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _on_training_start(self):
        # Initialize tqdm with total timesteps
        # mininterval=1.0 prevents it from trying to update faster than 1x per second
        self.pbar = tqdm(total=self.locals['total_timesteps'], mininterval=1.0, desc="Training")
        self.last_update_step = 0

    def _on_step(self) -> bool:
        # Only update the progress bar every `update_interval` steps, skipping the initial call at n_calls=0
        if self.n_calls > 0 and self.n_calls % self.update_interval == 0:
            # Calculate how many steps passed since last update (usually == update_interval)
            step_delta = self.num_timesteps - self.last_update_step

            if step_delta > 0:
                self.pbar.update(step_delta)
                self.pbar.set_description(f"Steps: {self.num_timesteps}")
                self.last_update_step = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        # Ensure the bar reaches 100% at the end
        if self.pbar is not None:
            remaining = self.locals['total_timesteps'] - self.last_update_step
            if remaining > 0:
                self.pbar.update(remaining)
            self.pbar.close()


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
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            thread_name = threading.current_thread().name
            handler = logging.FileHandler(f"{self.__class__.__name__.lower()}_{thread_name}.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    self._logger.info(f"Num timesteps: {self.num_timesteps}")
                    self._logger.info(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        self._logger.info(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting:
    1. Price vs EMA vs Action (Regime Analysis)
    2. Financial Metrics (Sharpe, Drawdown, Benchmark Comparison)
    """

    def __init__(self, verbose=0, buy_threshold=0.1, sell_threshold=-0.1):
        super().__init__(verbose)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.ep_prices = []
        self.ep_emas = []
        self.ep_actions = []
        self.ep_portfolio = []
        self.ep_dates = []
        self.ep_rewards = []
        # Reward components
        self.ep_reward_base = []
        self.ep_reward_fee = []
        self.ep_reward_action_change = []
        self.ep_reward_trend = []
        self.ep_reward_holding = []
        self.ep_reward_inertia = []
        self.ep_reward_closer = []
        self.ep_reward_overtrade = []
        self.ep_reward_episode = []
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            thread_name = threading.current_thread().name
            handler = logging.FileHandler(f"{self.__class__.__name__.lower()}_{thread_name}.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

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
        # Use the clipped action from info instead of raw from locals
        action = info.get('action', 0)

        current_price = info.get('current_price', info.get('price', 0))
        ema_50 = info.get('ema50', 0)
        portfolio_value = info.get('net_worth', 0)
        current_date = info.get('timestamp', f"{self.n_calls}")

        # Get reward
        rewards = self.locals['rewards']
        reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards

        # 3. Store
        self.ep_prices.append(current_price)
        self.ep_emas.append(ema_50)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        # Store reward components
        self.ep_reward_base.append(info.get('reward_base', 0.0))
        self.ep_reward_fee.append(info.get('reward_fee', 0.0))
        self.ep_reward_action_change.append(info.get('reward_action_change', 0.0))
        self.ep_reward_trend.append(info.get('reward_trend', 0.0))
        self.ep_reward_holding.append(info.get('reward_holding', 0.0))
        self.ep_reward_inertia.append(info.get('reward_inertia', 0.0))
        self.ep_reward_closer.append(info.get('reward_closer', 0.0))
        self.ep_reward_overtrade.append(info.get('reward_overtrade', 0.0))
        self.ep_reward_episode.append(info.get('reward_episode', 0.0))
        self.ep_portfolio.append({
            'step': self.n_calls,
            'net_worth': portfolio_value,
            'price': current_price,
            'action': action,
            'shares': info.get('shares_held', 0),
            'exposure': (info.get('shares_held', 0) * current_price) / portfolio_value if portfolio_value != 0 else 0.0,
            'trade_executed': info.get('trade_executed', False),
            'panic_close': info.get('panic_close', False)
        })
        self.ep_dates.append(current_date)

        # 4. Episode End Logic
        dones = self.locals['dones']
        is_done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

        if is_done:
            self._calculate_financial_metrics() # <--- NEW: Calculate Stats
            # self._plot_regime_chart()  # Disabled to avoid mismatch with logs from subprocess training
            
            # Reset
            self.ep_prices = []
            self.ep_emas = []
            self.ep_actions = []
            self.ep_portfolio = []
            self.ep_dates = []
            self.ep_rewards = []
            # Reset reward components
            self.ep_reward_base = []
            self.ep_reward_fee = []
            self.ep_reward_action_change = []
            self.ep_reward_trend = []
            self.ep_reward_holding = []
            self.ep_reward_inertia = []
            self.ep_reward_closer = []
            self.ep_reward_overtrade = []
            self.ep_reward_episode = []

        return True

    def _calculate_financial_metrics(self):
        """Calculates Sharpe, Sortino, Drawdown, Calmar, and Benchmark comparison."""
        self._logger.info(f"_calculate_financial_metrics called with len(ep_portfolio)={len(self.ep_portfolio)}")
        if len(self.ep_portfolio) < 2:
            self._logger.info("Skipping financial metrics calculation: insufficient portfolio data")
            return

        # Convert lists to arrays
        portfolio = np.array([point['net_worth'] for point in self.ep_portfolio])
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

        self._logger.info(f"Calculated metrics: sharpe={sharpe}, sortino={sortino}, max_drawdown={max_drawdown}, annualized_return={annualized_return}")

        # 5. Log to WandB
        try:
            if wandb.run is not None:
                self._logger.info(f"Logging financial metrics to wandb at global_step={self.num_timesteps}")
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
                    "financial/final_networth": portfolio[-1],
                    "global_step": self.num_timesteps
                })
                self._logger.info("Successfully logged financial metrics to wandb")
            else:
                self._logger.warning("wandb.run is None, skipping financial metrics log")
        except Exception as e:
            self._logger.error(f"Failed to log financial metrics to wandb: {e}")

    def _plot_regime_chart(self):
        if len(self.ep_prices) < 10:
            return
        self._logger.info("_plot_regime_chart called")

        import numpy as np
        import matplotlib.pyplot as plt

        steps   = np.arange(len(self.ep_prices))
        prices  = np.array(self.ep_prices, dtype=float)
        emas    = np.array(self.ep_emas,   dtype=float)
        actions = np.array(self.ep_actions, dtype=float)
        rewards = np.array(self.ep_rewards, dtype=float)
        portfolio = np.array([point['net_worth'] for point in self.ep_portfolio], dtype=float)
        dates   = self.ep_dates

        # Reward components arrays
        reward_base = np.array(self.ep_reward_base, dtype=float)
        reward_fee = np.array(self.ep_reward_fee, dtype=float)
        reward_action_change = np.array(self.ep_reward_action_change, dtype=float)
        reward_trend = np.array(self.ep_reward_trend, dtype=float)
        reward_holding = np.array(self.ep_reward_holding, dtype=float)
        reward_inertia = np.array(self.ep_reward_inertia, dtype=float)
        reward_closer = np.array(self.ep_reward_closer, dtype=float)
        reward_overtrade = np.array(self.ep_reward_overtrade, dtype=float)
        reward_episode = np.array(self.ep_reward_episode, dtype=float)

        # === DEBUG: Log reward component statistics ===
        self._logger.info("=== REWARD COMPONENT ANALYSIS ===")
        component_names = ['Base', 'Fee', 'Action_Change', 'Trend', 'Holding', 'Inertia', 'Closer', 'Overtrade', 'Episode']
        for name, component_array in zip(component_names, [reward_base, reward_fee, reward_action_change, reward_trend, reward_holding, reward_inertia, reward_closer, reward_overtrade, reward_episode]):
            self._logger.info(f"{name}: min={np.min(component_array):.6f}, max={np.max(component_array):.6f}, mean={np.mean(component_array):.6f}, std={np.std(component_array):.6f}")
            negative_count = np.sum(component_array < 0)
            positive_count = np.sum(component_array > 0)
            zero_count = np.sum(component_array == 0)
            self._logger.info(f"  Negative: {negative_count}, Positive: {positive_count}, Zero: {zero_count}")
        self._logger.info("===============================")
        # ==========================================

        # === 1. INSERT DEBUG COUNTERS HERE ===
        total_steps = len(self.ep_portfolio)
        total_plotted_buys = 0
        total_plotted_sells = 0
        executed_true_count = sum(1 for p in self.ep_portfolio if p.get('trade_executed', False))
        
        # DEBUG: Analyze action distribution
        actions_array = np.array(self.ep_actions, dtype=float)
        positive_actions = np.sum(actions_array > 0)
        negative_actions = np.sum(actions_array < 0)
        zero_actions = np.sum(actions_array == 0)
        
        # Show sample of actions for debugging
        sample_actions = actions_array[:min(20, len(actions_array))]
        self._logger.info(f"Action distribution - Positive: {positive_actions}, Negative: {negative_actions}, Zero: {zero_actions}")
        self._logger.info(f"Sample actions: {sample_actions}")

        # Log the raw data status
        self._logger.info("--- CHART DEBUG ---")
        self._logger.info(f"Total Steps recorded: {total_steps}")
        self._logger.info(f"Steps with 'trade_executed'=True: {executed_true_count}")
        # =====================================

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, figsize=(27, 12), sharex=True,
            gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]}
        )

        # Price + EMA + bull/bear shading
        ax1.plot(steps, prices, label='Price', color='black', linewidth=1.2)
        if np.any(emas > 0):
            ax1.plot(steps, emas, label='EMA 50', color='orange', linestyle='--', linewidth=1)
            min_len = min(len(prices), len(emas))
            ax1.fill_between(
                steps[:min_len], prices[:min_len], emas[:min_len],
                where=(prices[:min_len] > emas[:min_len]),
                color='green', alpha=0.1, label='Bull'
            )
            ax1.fill_between(
                steps[:min_len], prices[:min_len], emas[:min_len],
                where=(prices[:min_len] <= emas[:min_len]),
                color='red', alpha=0.1, label='Bear'
            )

        # Markers for Buy/Sell (ONLY IF EXECUTED)
        # We check point.get('trade_executed', False) to use the flag.
        buy_label_used = False
        sell_label_used = False
        panic_label_used = False
        for i, point in enumerate(self.ep_portfolio):
            is_executed = point.get('trade_executed', False)
            is_panic = point.get('panic_close', False)
            if is_executed:
                action_val = point['action']
                if action_val > 0:  # Buy
                    total_plotted_buys += 1  # <--- Count Buy
                    ax1.scatter(
                        steps[i], prices[i],
                        color='green', marker='^', s=50,
                        label='Buy' if not buy_label_used else ""
                    )
                    buy_label_used = True
                elif action_val < 0:  # Sell
                    total_plotted_sells += 1  # <--- Count Sell
                    ax1.scatter(
                        steps[i], prices[i],
                        color='red', marker='v', s=50,
                        label='Sell' if not sell_label_used else ""
                    )
                    sell_label_used = True
            elif is_panic:
                # Panic close marker
                ax1.scatter(
                    steps[i], prices[i],
                    color='orange', marker='X', s=100,
                    label='Panic Close' if not panic_label_used else ""
                )
                panic_label_used = True

        last_pv = self.ep_portfolio[-1]['net_worth'] if self.ep_portfolio else 0.0
        ax1.set_title(f"Thread 0 | PV: {last_pv:.2f}")
        ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc='lower left')
        ax1.grid(True, alpha=0.3)

        # --- Action bar plot: color = long/short, height = exposure ---
        # If actions are in [-max_leverage, +max_leverage], this shows exposure directly.
        colors = ['green' if a >= 0 else 'red' for a in actions]
        ax2.bar(steps, actions, color=colors, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Action")
        ax2.grid(True, axis='y', alpha=0.3)

        # Networth subplot
        ax3.plot(steps, portfolio, label='Networth', color='blue', linewidth=1.5)
        ax3.set_ylabel("Networth")
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # Rewards subplot - line plot for each component (no stacking)
        component_arrays = [
            reward_base,
            reward_fee,
            reward_action_change,
            reward_trend,
            reward_holding,
            reward_inertia,
            reward_closer,
            reward_overtrade,
            reward_episode
        ]
        component_labels = [
            'Base (net worth)',
            'Fee penalty',
            'Action change penalty',
            'Trend alignment',
            'Holding cost',
            'Inertia penalty',
            'Closer bonus',
            'Overtrading penalty',
            'Episode termination'
        ]
        # Colors from tab20c colormap
        colors = plt.cm.tab20c(np.linspace(0, 1, len(component_arrays)))
        
        # Plot each component as a separate line (no stacking)
        for i, (component_array, label, color) in enumerate(zip(component_arrays, component_labels, colors)):
            ax4.plot(steps, component_array, label=label, color=color, linewidth=1, alpha=0.8)
        
        # Add horizontal line at zero for reference
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax4.set_ylabel("Reward Components")
        ax4.grid(True, alpha=0.3)
        # Legend with small font, placed below
        ax4.legend(bbox_to_anchor=(0, -0.25, 1, 0.1), loc='upper center', fontsize='xx-small', ncol=6, framealpha=0.7)

        # Calculate and plot total cumulative reward in separate subplot
        total_reward = np.sum(component_arrays, axis=0)
        ax5.plot(steps, total_reward, color='black', linewidth=2, alpha=0.9, label='Total Reward', zorder=5)
        
        # Add horizontal line at zero for reference
        ax5.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Add annotations for total reward at key points (every 10th step)
        if len(steps) > 10:
            # Annotate every 10th step to avoid clutter
            for i in range(0, len(steps), max(1, len(steps)//8)):
                step_idx = min(i, len(steps)-1)
                ax5.annotate(f'{total_reward[step_idx]:.2f}',
                           (steps[step_idx], total_reward[step_idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8, ha='left')
        else:
            # Annotate all steps if not too many
            for i, (step, reward) in enumerate(zip(steps, total_reward)):
                ax5.annotate(f'{reward:.2f}', (step, reward),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8, ha='left')
        
        ax5.set_ylabel("Total Reward")
        ax5.grid(True, alpha=0.3)
        # Legend for total reward
        ax5.legend(loc='upper right', fontsize='small')

        # Date labels (move to bottom subplot)
        num_ticks = min(8, len(steps))
        tick_indices = np.linspace(0, len(steps) - 1, num_ticks, dtype=int)
        tick_labels = []
        for idx in tick_indices:
            raw_date = dates[idx]
            if hasattr(raw_date, 'strftime'):
                d_str = raw_date.strftime('%Y-%m-%d\n%H:%M')
            else:
                d_str = str(raw_date)[:16]
            tick_labels.append(d_str)

        ax5.set_xticks(tick_indices)
        ax5.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=8)
        ax5.set_xlabel("Time Steps")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(hspace=0.4)

        # === 3. LOG THE FINAL COUNT ===
        self._logger.info(f"Chart Markers Plotted -> Buys: {total_plotted_buys}, Sells: {total_plotted_sells}")
        self._logger.info("-------------------")
        # ==============================

        try:
            if wandb.run is not None:
                self._logger.info(f"Attempting to log chart to WandB, wandb.run.id: {wandb.run.id}")
                wandb.log({"trade_analysis/thread_0_chart": wandb.Image(fig)})
                self._logger.info("Successfully logged regime chart to WandB")
            else:
                self._logger.warning("wandb.run is None, skipping chart log")
        except Exception as e:
            self._logger.error(f"Failed to log regime chart to WandB: {e}")
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
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            thread_name = threading.current_thread().name
            handler = logging.FileHandler(f"{self.__class__.__name__.lower()}_{thread_name}.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        self._logger.info(f"CustomEvalCallback eval_env type: {type(self.eval_env)}")

        # For chart plotting
        self.ep_prices = []
        self.ep_emas = []
        self.ep_actions = []
        self.ep_portfolio = []
        self.ep_dates = []
        self.ep_rewards = []
        # Reward components
        self.ep_reward_base = []
        self.ep_reward_fee = []
        self.ep_reward_action_change = []
        self.ep_reward_trend = []
        self.ep_reward_holding = []
        self.ep_reward_inertia = []
        self.ep_reward_closer = []
        self.ep_reward_overtrade = []
        self.ep_reward_episode = []

    def _evaluate_with_portfolio(self):
        """
        Evaluate the agent using the portfolio environment.
        FIXED: Handles VecEnv vs Gym API and adds Progress Bar.
        """
        self._logger.info("Starting _evaluate_with_portfolio")

        # 1. Reset Environment (Handle SB3 VecEnv returning only obs)
        reset_result = self.eval_env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # Gymnasium API
        else:
            obs = reset_result     # SB3 VecEnv API

        # Initialize tracking
        episode_rewards = []
        portfolio_values = []
        trades_per_episode = []

        # 2. Add Progress Bar (tqdm) to see status during "block"
        ep_bar = tqdm(range(self.n_eval_episodes), desc="Evaluating Portfolio", unit="ep")

        for episode_idx in ep_bar:
            done = False
            episode_reward = 0.0
            step_count = 0
            step_bar = tqdm(desc="Steps", unit="step", leave=False)

            # Reset episode data
            if episode_idx == self.n_eval_episodes - 1:  # Collect data for last episode
                self.ep_prices = []
                self.ep_emas = []
                self.ep_actions = []
                self.ep_portfolio = []
                self.ep_dates = []
                self.ep_rewards = []
                # Reset reward components
                self.ep_reward_base = []
                self.ep_reward_fee = []
                self.ep_reward_action_change = []
                self.ep_reward_trend = []
                self.ep_reward_holding = []
                self.ep_reward_inertia = []
                self.ep_reward_closer = []
                self.ep_reward_overtrade = []
                self.ep_reward_episode = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

                # Handle Step API (VecEnv returns 4 items, Gymnasium returns 5)
                step_result = self.eval_env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated, truncated = done, done
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                # Unpack Reward
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0]

                # Unpack Info (VecEnv returns a list of dicts)
                if isinstance(info, list):
                    info = info[0]

                # Collect data for last episode
                if episode_idx == self.n_eval_episodes - 1:
                    current_price = info.get('current_price', info.get('price', 0))
                    ema_50 = info.get('ema50', 0)
                    portfolio_value = info.get('net_worth', 0)
                    current_date = info.get('timestamp', f"{step_count}")
                    
                    # FIX: Use consistent action extraction like TensorboardCallback
                    # Always use the processed action from environment info, no fallback needed
                    action_val = info.get('action', 0)

                    self.ep_prices.append(current_price)
                    self.ep_emas.append(ema_50)
                    self.ep_actions.append(action_val)
                    self.ep_rewards.append(reward)
                    # Store reward components
                    self.ep_reward_base.append(info.get('reward_base', 0.0))
                    self.ep_reward_fee.append(info.get('reward_fee', 0.0))
                    self.ep_reward_action_change.append(info.get('reward_action_change', 0.0))
                    self.ep_reward_trend.append(info.get('reward_trend', 0.0))
                    self.ep_reward_holding.append(info.get('reward_holding', 0.0))
                    self.ep_reward_inertia.append(info.get('reward_inertia', 0.0))
                    self.ep_reward_closer.append(info.get('reward_closer', 0.0))
                    self.ep_reward_overtrade.append(info.get('reward_overtrade', 0.0))
                    self.ep_reward_episode.append(info.get('reward_episode', 0.0))
                    self.ep_portfolio.append({
                        'step': step_count,
                        'net_worth': portfolio_value,
                        'price': current_price,
                        'action': action_val,
                        'shares': info.get('shares_held', 0),
                        'trade_executed': info.get('trade_executed', False),
                        'panic_close': info.get('panic_close', False)
                    })
                    self.ep_dates.append(current_date)

                episode_reward += reward
                step_count += 1
                step_bar.update(1)
                step_bar.set_postfix(reward=episode_reward)

            episode_rewards.append(episode_reward)

            # FIX: Check for 'net_worth' OR 'portfolio_value'
            current_val = info.get('net_worth', info.get('portfolio_value', 0.0))

            # If we still get 0, fall back to initial balance + reward
            if current_val == 0 and self.initial_balance:
                  current_val = self.initial_balance + episode_reward

            portfolio_values.append(current_val)
            trades_per_episode.append(info.get('trades_per_episode', 0))
            ep_bar.update(1)

            # Display explicit Valuation
            ep_bar.set_postfix(NetWorth=f"${np.mean(portfolio_values):.2f}")

            # Reset for next episode
            reset_result = self.eval_env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result

        # Calculate averages
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_portfolio = np.mean(portfolio_values)
        std_portfolio = np.std(portfolio_values)
        mean_trades = np.mean(trades_per_episode)

        return mean_reward, std_reward, mean_portfolio, std_portfolio, mean_trades

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            # Diagnostic logs for mangled output
            import shutil
            terminal_width = shutil.get_terminal_size().columns
            self._logger.info(f"Evaluation triggered at n_calls={self.n_calls}, num_timesteps={self.num_timesteps}, eval_freq={self.eval_freq}, terminal_width={terminal_width}")
            self._logger.info(f"About to log eval metrics - checking for output interference")
            # --- UPDATED: Phase Switching ---
            if self.num_timesteps % 250000 == 0:
                self._logger.info("Starting phase switching")
                current_phase = getattr(self.model.env, 'phase', 1)
                new_phase = min(current_phase + 1, 3)  # Up to phase 3
                # Broadcast to train env (works for Subproc via attr access)
                if hasattr(self.model.env, 'set_attr'):
                    self.model.env.set_attr('phase', new_phase)
                else:
                    # Fallback: Set on wrapped env
                    self.model.env.phase = new_phase
                if wandb.run is not None:
                    wandb.log({'curriculum/phase': new_phase, 'global_step': self.num_timesteps})
                self._logger.info(f"Phase switching completed, new_phase={new_phase}")

            # Log thresholds
            try:
                buy_thresh = self.eval_env.get_attr('buy_threshold')[0]
                sell_thresh = self.eval_env.get_attr('sell_threshold')[0]
                self._logger.info(f"Current thresholds: buy={buy_thresh}, sell={sell_thresh}")
            except:
                self._logger.info("Could not retrieve thresholds")

            self._logger.info("Starting evaluation")
            try:
                mean_reward, std_reward, mean_portfolio, std_portfolio, mean_trades = self._evaluate_with_portfolio()
                self._logger.info(f"Evaluation completed, mean_reward={mean_reward}, mean_portfolio={mean_portfolio}, trades_per_episode={mean_trades}")
            except Exception as e:
                self._logger.error(f"Evaluation failed with exception: {e}")
                import traceback
                self._logger.error(traceback.format_exc())
                return True
            if self.log_path is not None:
                self._logger.info(f"Recording eval metrics: mean_reward={mean_reward}, mean_portfolio={mean_portfolio}")
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                self.logger.record("eval/mean_portfolio", mean_portfolio)
                self.logger.record("eval/std_portfolio", std_portfolio)
                self.logger.record("eval/trades_per_episode", mean_trades)
                self._logger.info("Eval metrics recorded to SB3 logger")

            # Log to WandB if available
            if wandb.run is not None:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/mean_portfolio": mean_portfolio,
                    "eval/std_portfolio": std_portfolio,
                    "eval/trades_per_episode": mean_trades,
                    "global_step": self.num_timesteps
                })
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_std_reward = std_reward
                self.best_mean_portfolio = mean_portfolio
                self.best_std_portfolio = std_portfolio
                self.best_mean_trades = mean_trades
                self._logger.info(f"New best model found! Mean Reward: {mean_reward}, Mean Portfolio: {mean_portfolio}, Trades per Episode: {mean_trades}")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                    self._logger.info(f"Best model saved to {self.best_model_save_path}")
                self._log_best_to_wandb(mean_reward, std_reward)

            # Plot regime chart for evaluation
            if len(self.ep_prices) > 10:
                self._plot_regime_chart()

        return True

    def _log_best_to_wandb(self, mean_reward, std_reward):
        if wandb.run is not None:
            wandb.log({
                "best_eval/mean_reward": mean_reward,
                "best_eval/std_reward": std_reward,
                "best_eval/mean_portfolio": self.best_mean_portfolio,
                "best_eval/std_portfolio": self.best_std_portfolio,
                "best_eval/trades_per_episode": self.best_mean_trades,
                "global_step": self.num_timesteps
            })

    def _plot_regime_chart(self):
        if len(self.ep_prices) < 10:
            return
        self._logger.info("_plot_regime_chart called for evaluation")

        import numpy as np
        import matplotlib.pyplot as plt

        steps   = np.arange(len(self.ep_prices))
        prices  = np.array(self.ep_prices, dtype=float)
        emas    = np.array(self.ep_emas,   dtype=float)
        actions = np.array(self.ep_actions, dtype=float)
        rewards = np.array(self.ep_rewards, dtype=float)
        portfolio = np.array([point['net_worth'] for point in self.ep_portfolio], dtype=float)
        dates   = self.ep_dates

        # Reward components arrays
        reward_base = np.array(self.ep_reward_base, dtype=float)
        reward_fee = np.array(self.ep_reward_fee, dtype=float)
        reward_action_change = np.array(self.ep_reward_action_change, dtype=float)
        reward_trend = np.array(self.ep_reward_trend, dtype=float)
        reward_holding = np.array(self.ep_reward_holding, dtype=float)
        reward_inertia = np.array(self.ep_reward_inertia, dtype=float)
        reward_closer = np.array(self.ep_reward_closer, dtype=float)
        reward_overtrade = np.array(self.ep_reward_overtrade, dtype=float)
        reward_episode = np.array(self.ep_reward_episode, dtype=float)

        # === DEBUG: Log reward component statistics ===
        self._logger.info("=== EVALUATION REWARD COMPONENT ANALYSIS ===")
        component_names = ['Base', 'Fee', 'Action_Change', 'Trend', 'Holding', 'Inertia', 'Closer', 'Overtrade', 'Episode']
        for name, component_array in zip(component_names, [reward_base, reward_fee, reward_action_change, reward_trend, reward_holding, reward_inertia, reward_closer, reward_overtrade, reward_episode]):
            self._logger.info(f"EVAL {name}: min={np.min(component_array):.6f}, max={np.max(component_array):.6f}, mean={np.mean(component_array):.6f}, std={np.std(component_array):.6f}")
            negative_count = np.sum(component_array < 0)
            positive_count = np.sum(component_array > 0)
            zero_count = np.sum(component_array == 0)
            self._logger.info(f"  Negative: {negative_count}, Positive: {positive_count}, Zero: {zero_count}")
        self._logger.info("===========================================")
        # ==========================================

        # === 1. INSERT DEBUG COUNTERS HERE ===
        total_steps = len(self.ep_portfolio)
        total_plotted_buys = 0
        total_plotted_sells = 0
        executed_true_count = sum(1 for p in self.ep_portfolio if p.get('trade_executed', False))
        
        # DEBUG: Analyze action distribution in evaluation
        actions_array = np.array(self.ep_actions, dtype=float)
        positive_actions = np.sum(actions_array > 0)
        negative_actions = np.sum(actions_array < 0)
        zero_actions = np.sum(actions_array == 0)
        
        # Show sample of actions for debugging
        sample_actions = actions_array[:min(20, len(actions_array))]
        self._logger.info(f"EVAL Action distribution - Positive: {positive_actions}, Negative: {negative_actions}, Zero: {zero_actions}")
        self._logger.info(f"EVAL Sample actions: {sample_actions}")

        # Log the raw data status
        self._logger.info("--- EVALUATION CHART DEBUG ---")
        self._logger.info(f"Total Steps recorded: {total_steps}")
        self._logger.info(f"Steps with 'trade_executed'=True: {executed_true_count}")
        # =====================================

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, figsize=(27, 12), sharex=True,
            gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]}
        )

        # Price + EMA + bull/bear shading
        ax1.plot(steps, prices, label='Price', color='black', linewidth=1.2)
        if np.any(emas > 0):
            ax1.plot(steps, emas, label='EMA 50', color='orange', linestyle='--', linewidth=1)
            min_len = min(len(prices), len(emas))
            ax1.fill_between(
                steps[:min_len], prices[:min_len], emas[:min_len],
                where=(prices[:min_len] > emas[:min_len]),
                color='green', alpha=0.1, label='Bull'
            )
            ax1.fill_between(
                steps[:min_len], prices[:min_len], emas[:min_len],
                where=(prices[:min_len] <= emas[:min_len]),
                color='red', alpha=0.1, label='Bear'
            )

        # Markers for Buy/Sell (ONLY IF EXECUTED)
        # We check point.get('trade_executed', False) to use the flag.
        buy_label_used = False
        sell_label_used = False
        panic_label_used = False
        for i, point in enumerate(self.ep_portfolio):
            is_executed = point.get('trade_executed', False)
            if is_executed:
                action_val = point['action']
                if action_val > 0:  # Buy
                    total_plotted_buys += 1  # <--- Count Buy
                    ax1.scatter(
                        steps[i], prices[i],
                        color='green', marker='^', s=50,
                        label='Buy' if not buy_label_used else ""
                    )
                    buy_label_used = True
                elif action_val < 0:  # Sell
                    total_plotted_sells += 1  # <--- Count Sell
                    ax1.scatter(
                        steps[i], prices[i],
                        color='red', marker='v', s=50,
                        label='Sell' if not sell_label_used else ""
                    )
                    sell_label_used = True


        last_pv = self.ep_portfolio[-1]['net_worth'] if self.ep_portfolio else 0.0
        episode = self.n_eval_episodes
        step = len(self.ep_portfolio)
        global_step = self.num_timesteps
        ax1.set_title(f"Evaluation | Episode: {episode} | Step: {step} | Global Step: {global_step} | PV: {last_pv:.2f}")
        ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc='lower left')
        ax1.grid(True, alpha=0.3)

        # --- Action bar plot: color = long/short, height = exposure ---
        # If actions are in [-max_leverage, +max_leverage], this shows exposure directly.
        colors = ['green' if a >= 0 else 'red' for a in actions]
        ax2.bar(steps, actions, color=colors, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Actions")
        ax2.grid(True, axis='y', alpha=0.3)

        # Networth subplot
        ax3.plot(steps, portfolio, label='Networth', color='blue', linewidth=1.5)
        ax3.set_ylabel("Networth")
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # Rewards subplot - line plot for each component (no stacking)
        component_arrays = [
            reward_base,
            reward_fee,
            reward_action_change,
            reward_trend,
            reward_holding,
            reward_inertia,
            reward_closer,
            reward_overtrade,
            reward_episode
        ]
        component_labels = [
            'Base (net worth)',
            'Fee penalty',
            'Action change penalty',
            'Trend alignment',
            'Holding cost',
            'Inertia penalty',
            'Closer bonus',
            'Overtrading penalty',
            'Episode termination'
        ]
        # Colors from tab20c colormap
        colors = plt.cm.tab20c(np.linspace(0, 1, len(component_arrays)))
        
        # Plot each component as a separate line (no stacking)
        for i, (component_array, label, color) in enumerate(zip(component_arrays, component_labels, colors)):
            ax4.plot(steps, component_array, label=label, color=color, linewidth=1, alpha=0.8)
        
        # Add horizontal line at zero for reference
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax4.set_ylabel("Reward Components")
        ax4.grid(True, alpha=0.3)
        # Legend with small font, placed below
        ax4.legend(bbox_to_anchor=(0, -0.25, 1, 0.1), loc='upper center', fontsize='xx-small', ncol=6, framealpha=0.7)

        # Calculate and plot total cumulative reward in separate subplot
        total_reward = np.sum(component_arrays, axis=0)
        ax5.plot(steps, total_reward, color='black', linewidth=2, alpha=0.9, label='Total Reward', zorder=5)
        
        # Add horizontal line at zero for reference
        ax5.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Add annotations for total reward at key points (every 10th step)
        if len(steps) > 10:
            # Annotate every 10th step to avoid clutter
            for i in range(0, len(steps), max(1, len(steps)//8)):
                step_idx = min(i, len(steps)-1)
                ax5.annotate(f'{total_reward[step_idx]:.2f}',
                           (steps[step_idx], total_reward[step_idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8, ha='left')
        else:
            # Annotate all steps if not too many
            for i, (step, reward) in enumerate(zip(steps, total_reward)):
                ax5.annotate(f'{reward:.2f}', (step, reward),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8, ha='left')
        
        ax5.set_ylabel("Total Reward")
        ax5.grid(True, alpha=0.3)
        # Legend for total reward
        ax5.legend(loc='upper right', fontsize='small')

        # Date labels (move to bottom subplot)
        num_ticks = min(8, len(steps))
        tick_indices = np.linspace(0, len(steps) - 1, num_ticks, dtype=int)
        tick_labels = []
        for idx in tick_indices:
            raw_date = dates[idx]
            if hasattr(raw_date, 'strftime'):
                d_str = raw_date.strftime('%Y-%m-%d\n%H:%M')
            else:
                d_str = str(raw_date)[:16]
            tick_labels.append(d_str)

        ax5.set_xticks(tick_indices)
        ax5.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=8)
        ax5.set_xlabel("Time Steps")

        plt.tight_layout()

        # === 3. LOG THE FINAL COUNT ===
        self._logger.info(f"Evaluation Chart Markers Plotted -> Buys: {total_plotted_buys}, Sells: {total_plotted_sells}")
        self._logger.info("-------------------")

        try:
            if wandb.run is not None:
                self._logger.info(f"Attempting to log evaluation chart to WandB, wandb.run.id: {wandb.run.id}")
                wandb.log({"eval/trade_analysis": wandb.Image(fig)})
                self._logger.info("Successfully logged evaluation regime chart to WandB")
            else:
                self._logger.warning("wandb.run is None, skipping evaluation chart log")
        except Exception as e:
            self._logger.error(f"Failed to log evaluation regime chart to WandB: {e}")
        plt.close(fig)