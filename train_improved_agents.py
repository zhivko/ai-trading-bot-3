import pandas as pd
import numpy as np
import matplotlib
import talib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from improved_trading_env import ImprovedTradingEnv
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SafeImageRecorderCallback(BaseCallback):
    """Enhanced callback for recording trading visualizations safely."""
    
    def __init__(self, verbose=0, render_freq=5000):
        super(SafeImageRecorderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        # Only run every render_freq steps to save speed
        if self.n_calls % self.render_freq == 0 and self.n_calls != self.last_logged_step:
            print(f"Debug: Attempting to log visualization at n_calls={self.n_calls}, num_timesteps={self.num_timesteps}")
            try:
                # Safely access environment and render
                env = self.training_env.envs[0]
                print(f"Debug: Environment has render: {hasattr(env.unwrapped, 'render')}")
                if hasattr(env.unwrapped, 'render') and hasattr(env.unwrapped.render, '__call__'):
                    print("Debug: Calling render(mode='rgb_array')")
                    img = env.unwrapped.render(mode='rgb_array')
                    print(f"Debug: Render returned img type: {type(img)}, shape: {img.shape if img is not None else 'None'}")
                    print(f"Debug: wandb.run is not None: {wandb.run is not None}")
                    if img is not None and wandb.run is not None:
                        print("Debug: Logging to wandb")
                        wandb.log({
                            "trading_visualization": wandb.Image(
                                img,
                                caption=f"Step {self.num_timesteps} - Trading Performance"
                            )
                        })
                        self.last_logged_step = self.n_calls
                        print(f"Logged visualization at step {self.num_timesteps}")
                    else:
                        print(f"Debug: Not logging - img is None: {img is None}, wandb.run is None: {wandb.run is None}")
                else:
                    print("Debug: Environment does not have render method")
            except Exception as e:
                print(f"Warning: Could not log visualization at step {self.num_timesteps}: {e}")

        return True


class EnhancedWandbCallback(BaseCallback):
    """Enhanced callback with comprehensive logging for multi-dimensional actions and reward components."""
    
    def __init__(self, verbose=0, eval_freq=10000):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_count = 0
        
        # Episode tracking
        self.ep_net_worths = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.ep_trade_count = 0
        self.ep_buy_count = 0
        self.ep_sell_count = 0
        self.ep_prices = []
        self.ep_actions = []
        self.ep_action_components = []
        self.ep_dates = []
        self.ep_balances = []
        self.ep_shares = []
        self.ep_reward_components = []
        self.ep_trade_statistics = []
        self.ep_market_regimes = []
        
        # Performance tracking
        self.best_net_worth = 0
        self.episode_returns = []
        
    def _on_step(self) -> bool:
        if wandb.run is None:
            return True
            
        # Log basic training metrics
        wandb.log({"timesteps": self.num_timesteps})
        
        # Extract environment information safely
        try:
            infos = self.locals.get('infos', [])
            if infos:
                info = infos[0] if isinstance(infos, list) else infos
                
                # Extract key metrics
                net_worth = info.get('net_worth', 0)
                current_price = info.get('current_price', info.get('price', 0))
                balance = info.get('balance', 0)
                shares_held = info.get('shares_held', 0)
                market_regime = info.get('market_regime', 'unknown')
                trade_executed = info.get('trade_executed', False)
                
                # Log current state
                wandb.log({
                    "current_net_worth": net_worth,
                    "current_price": current_price,
                    "balance": balance,
                    "shares_held": shares_held,
                    "market_regime": market_regime,
                    "trade_executed": trade_executed
                })
                
                # Track episode data
                self.ep_net_worths.append(net_worth)
                self.ep_prices.append(current_price)
                self.ep_balances.append(balance)
                self.ep_shares.append(shares_held)
                self.ep_market_regimes.append(market_regime)
                
                # Extract and log action components
                action_components = info.get('action_components', [0, 0, 0])
                if isinstance(action_components, (list, np.ndarray)) and len(action_components) >= 3:
                    self.ep_action_components.append(action_components)
                    self.ep_actions.append(action_components[0])  # Position target for basic tracking
                else:
                    self.ep_actions.append(0)
                
                # Extract and log reward components
                reward_components = info.get('reward_components', {})
                if reward_components:
                    self.ep_reward_components.append(reward_components)
                
                # Track trade statistics
                if trade_executed:
                    self.ep_trade_count += 1
                    # Analyze action for buy/sell classification
                    if len(action_components) > 0:
                        position_target = action_components[0]
                        if position_target > 0.1:
                            self.ep_buy_count += 1
                        elif position_target < -0.1:
                            self.ep_sell_count += 1
                
                # Track best performance
                if net_worth > self.best_net_worth:
                    self.best_net_worth = net_worth
                    wandb.log({"best_net_worth": self.best_net_worth})
                
        except Exception as e:
            print(f"Warning: Error logging step data: {e}")
        
        # Generate periodic visualizations
        if self.num_timesteps % (self.eval_freq // 2) == 0 and len(self.ep_balances) > 10:
            self._generate_comprehensive_plot()
        
        return True
    
    def _on_rollout_end(self):
        """Handle end of episode."""
        if wandb.run is None:
            return True
            
        self.episode_count += 1
        
        try:
            episode_return = 0.0
            # Calculate episode metrics
            if len(self.ep_net_worths) > 1:
                episode_return = (self.ep_net_worths[-1] - self.ep_net_worths[0]) / self.ep_net_worths[0]
                self.episode_returns.append(episode_return)

                # Calculate additional metrics
                max_drawdown = self._calculate_max_drawdown(self.ep_net_worths)
                sharpe_ratio = self._calculate_sharpe_ratio(self.episode_returns[-100:])  # Last 100 episodes

                # Log episode metrics
                wandb.log({
                    "episode_number": self.episode_count,
                    "episode_return": episode_return,
                    "episode_trade_count": self.ep_trade_count,
                    "episode_buy_count": self.ep_buy_count,
                    "episode_sell_count": self.ep_sell_count,
                    "episode_length": len(self.ep_net_worths),
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "cumulative_return": np.prod([r + 1 for r in self.episode_returns]) - 1
                })

            # Log reward component analysis
            if self.ep_reward_components:
                self._log_detailed_reward_analysis()

            # Log action component analysis
            if self.ep_action_components:
                self._log_detailed_action_analysis()

            # Generate episode summary visualization
            if len(self.ep_net_worths) > 10:
                self._generate_episode_summary_plot()

            if len(self.ep_net_worths) > 1:
                print(f"Episode {self.episode_count} completed: "
                      f"Return: {episode_return:.2%}, "
                      f"Trades: {self.ep_trade_count}, "
                      f"Net Worth: ${self.ep_net_worths[-1]:.2f}")
            
        except Exception as e:
            print(f"Warning: Error logging episode end: {e}")
        
        # Reset episode tracking
        self._reset_episode_tracking()
        
        return True
    
    def _calculate_max_drawdown(self, net_worths):
        """Calculate maximum drawdown from net worth series."""
        if len(net_worths) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(net_worths)
        drawdown = (net_worths - peak) / peak
        return np.min(drawdown)
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio from returns series."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def _log_detailed_reward_analysis(self):
        """Log detailed analysis of reward components."""
        if not self.ep_reward_components:
            return
        
        try:
            # Aggregate reward components across episode
            component_stats = {}
            for component in self.ep_reward_components:
                for key, value in component.items():
                    if key not in component_stats:
                        component_stats[key] = []
                    component_stats[key].append(value)
            
            # Log component statistics
            for component, values in component_stats.items():
                if values:
                    wandb.log({
                        f"reward_component_{component}_mean": np.mean(values),
                        f"reward_component_{component}_std": np.std(values),
                        f"reward_component_{component}_min": np.min(values),
                        f"reward_component_{component}_max": np.max(values),
                        f"reward_component_{component}_total": np.sum(values)
                    })
        except Exception as e:
            print(f"Warning: Error in reward analysis: {e}")
    
    def _log_detailed_action_analysis(self):
        """Log detailed analysis of action components."""
        if not self.ep_action_components:
            return
        
        try:
            # Convert to numpy array for easier analysis
            actions_array = np.array(self.ep_action_components)
            
            # Log component statistics
            for i in range(actions_array.shape[1]):
                component_values = actions_array[:, i]
                component_names = ['position_target', 'size_intensity', 'hold_preference']
                component_name = component_names[i] if i < len(component_names) else f'component_{i}'
                
                wandb.log({
                    f"action_{component_name}_mean": np.mean(component_values),
                    f"action_{component_name}_std": np.std(component_values),
                    f"action_{component_name}_min": np.min(component_values),
                    f"action_{component_name}_max": np.max(component_values)
                })
            
            # Log action stability (how much actions change)
            if actions_array.shape[0] > 1:
                action_changes = np.diff(actions_array, axis=0)
                for i, component_name in enumerate(['position_target', 'size_intensity', 'hold_preference']):
                    if i < action_changes.shape[1]:
                        changes = action_changes[:, i]
                        wandb.log({
                            f"action_{component_name}_change_std": np.std(changes),
                            f"action_{component_name}_change_mean": np.mean(np.abs(changes))
                        })
        except Exception as e:
            print(f"Warning: Error in action analysis: {e}")
    
    def _generate_comprehensive_plot(self):
        """Generate and log comprehensive performance visualization."""
        if len(self.ep_balances) < 10 or wandb.run is None:
            return

        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

            steps = np.arange(len(self.ep_balances))

            # Plot 1: Net Worth and Prices
            axes[0].plot(steps, self.ep_prices, label='Price', color='black', alpha=0.7)
            axes[0].set_ylabel('Price ($)')
            axes[0].set_title(f'Enhanced Trading Performance - Step {self.num_timesteps}')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            # Plot 2: Portfolio Value
            axes[1].plot(steps, self.ep_balances, label='Cash Balance', color='blue', alpha=0.7)
            if hasattr(self, 'ep_net_worths') and len(self.ep_net_worths) == len(steps):
                axes[1].plot(steps, self.ep_net_worths, label='Net Worth', color='green', alpha=0.7)
            axes[1].set_ylabel('Value ($)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            # Plot 3: Position and Shares
            axes[2].plot(steps, self.ep_shares, label='Shares Held', color='purple', alpha=0.7)
            axes[2].set_ylabel('Shares')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            # Plot 4: Action Components
            if self.ep_action_components:
                actions_array = np.array(self.ep_action_components)
                axes[3].plot(steps, actions_array[:, 0], label='Position Target', color='red', alpha=0.7)
                if actions_array.shape[1] > 1:
                    axes[3].plot(steps, actions_array[:, 1], label='Size Intensity', color='orange', alpha=0.7)
                if actions_array.shape[1] > 2:
                    axes[3].plot(steps, actions_array[:, 2], label='Hold Preference', color='brown', alpha=0.7)
                axes[3].set_ylabel('Action Components')
                axes[3].legend()
            else:
                axes[3].plot(steps, self.ep_actions, label='Actions', color='red', alpha=0.7)
                axes[3].set_ylabel('Actions')
                axes[3].legend()

            axes[3].set_xlabel('Steps')
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()

            # Log to wandb
            wandb.log({"comprehensive_performance": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Error generating comprehensive plot: {e}")
    
    def _generate_episode_summary_plot(self):
        """Generate episode summary visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot 1: Net Worth Evolution
            steps = np.arange(len(self.ep_net_worths))
            axes[0, 0].plot(steps, self.ep_net_worths, color='green', linewidth=2)
            axes[0, 0].set_title('Episode Net Worth Evolution')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Net Worth ($)')
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Trade Distribution
            trade_types = ['Buy', 'Sell', 'Hold']
            trade_counts = [self.ep_buy_count, self.ep_sell_count,
                          max(0, len(self.ep_actions) - self.ep_buy_count - self.ep_sell_count)]
            colors = ['green', 'red', 'gray']
            axes[0, 1].bar(trade_types, trade_counts, color=colors, alpha=0.7)
            axes[0, 1].set_title('Trade Distribution')
            axes[0, 1].set_ylabel('Count')

            # Plot 3: Reward Components
            if self.ep_reward_components:
                reward_df = pd.DataFrame(self.ep_reward_components)
                for col in reward_df.columns:
                    axes[1, 0].plot(range(len(reward_df)), reward_df[col], label=col, alpha=0.7)
                axes[1, 0].set_title('Reward Components Over Episode')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Reward Value')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Market Regime Distribution
            if self.ep_market_regimes:
                regime_counts = pd.Series(self.ep_market_regimes).value_counts()
                axes[1, 1].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
                axes[1, 1].set_title('Market Regime Distribution')

            plt.tight_layout()

            # Log to wandb
            wandb.log({"episode_summary": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Error generating episode summary: {e}")
    
    def _reset_episode_tracking(self):
        """Reset all episode tracking variables."""
        self.ep_net_worths = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.ep_trade_count = 0
        self.ep_buy_count = 0
        self.ep_sell_count = 0
        self.ep_prices = []
        self.ep_actions = []
        self.ep_action_components = []
        self.ep_dates = []
        self.ep_balances = []
        self.ep_shares = []
        self.ep_reward_components = []
        self.ep_trade_statistics = []
        self.ep_market_regimes = []


class EvaluationCallback(BaseCallback):
    """Callback for periodic evaluation during training."""
    
    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=5, verbose=1):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n=== Evaluation at step {self.num_timesteps} ===")
            
            try:
                # Evaluate current policy
                mean_reward, std_reward = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True
                )
                
                # Log evaluation results
                if wandb.run is not None:
                    wandb.log({
                        "eval_mean_reward": mean_reward,
                        "eval_std_reward": std_reward,
                        "eval_timestep": self.num_timesteps
                    })
                
                print(f"Evaluation Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"New best reward: {mean_reward:.2f}")
                    self.model.save("best_sac_trading_model")
                    
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
        
        return True


def create_improved_data(df):
    """Enhanced data preparation with technical indicators."""
    # Calculate technical indicators using TA-Lib
    if 'ema_200' not in df.columns:
        df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
        macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        fastk_3_14, fastd_3_14 = talib.STOCHRSI(df['close'].values, timeperiod=14, fastk_period=3, fastd_period=3)
        df['sto_rsi_3_14'] = fastk_3_14
        fastk_10_60, fastd_10_60 = talib.STOCHRSI(df['close'].values, timeperiod=60, fastk_period=10, fastd_period=3)
        df['sto_rsi_10_60'] = fastk_10_60

    return df


def make_improved_env(data_frame, window_size=35, initial_balance=10000):
    """Create improved trading environment with enhanced features."""
    return ImprovedTradingEnv(
        df=data_frame,
        initial_balance=initial_balance,
        window_size=window_size,
        trading_fee_rate=0.0015,    # 0.15% trading fee
        max_exposure=0.8,           # 80% maximum exposure
        optimal_hold_duration=24    # 24 hours optimal hold duration
    )


def main():
    """Enhanced main training function with comprehensive evaluation."""
    
    # Configuration
    DATA_FILE = 'BTCUSDT_data.csv'
    TIMESTEPS = 1000000  # Increased for better training
    WINDOW_SIZE = 35
    INITIAL_BALANCE = 10000
    
    print("🚀 Starting Enhanced SAC Training with Improved Environment")
    print("=" * 60)
    
    # Initialize WandB with enhanced configuration
    wandb.init(
        project="crypto-trading-rl-improved",
        name=f"enhanced-sac-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "SAC",
            "environment": "ImprovedTradingEnv",
            "timesteps": TIMESTEPS,
            "window_size": WINDOW_SIZE,
            "initial_balance": INITIAL_BALANCE,
            "trading_fee_rate": 0.0015,
            "max_exposure": 0.8,
            "optimal_hold_duration": 24,
            "n_envs": 4
        },
        tags=["enhanced", "sac", "multi-dimensional-actions", "comprehensive-logging"]
    )
    
    # Load and prepare data
    try:
        print("📊 Loading and preparing data...")
        df = pd.read_csv(DATA_FILE)
        df = create_improved_data(df)
        
        # Create train/test split
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        print(f"Data loaded: {len(df)} samples")
        print(f"Training data: {len(train_df)} samples")
        print(f"Test data: {len(test_df)} samples")
        
    except FileNotFoundError:
        print(f"❌ Error: {DATA_FILE} not found. Please ensure data file exists.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create environments
    print("\n🏗️ Creating environments...")
    
    def make_train_env():
        return make_improved_env(train_df, WINDOW_SIZE, INITIAL_BALANCE)
    
    # Vectorized environment for training
    train_env = make_vec_env(make_train_env, n_envs=4)
    
    # Separate evaluation environment
    eval_env = make_improved_env(test_df, WINDOW_SIZE, INITIAL_BALANCE)
    
    # Initialize SAC model with enhanced configuration
    print("\n🤖 Initializing SAC model...")
    sac_model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        buffer_size=1000000,
        learning_rate=3e-5,
        ent_coef='auto',
        gamma=0.99,
        tau=0.02,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # Larger network for complex action space
            activation_fn=torch.nn.ReLU
        ),
        device='auto'
    )
    
    print("✅ SAC model initialized successfully")
    print(f"Action space: {sac_model.action_space}")
    print(f"Observation space: {sac_model.observation_space}")
    
    # Initialize callbacks
    print("\n📊 Setting up callbacks...")
    
    # Safe image recorder (less frequent to avoid performance issues)
    img_callback = SafeImageRecorderCallback(render_freq=5000)
    
    # Enhanced WandB callback with comprehensive logging
    wandb_callback = EnhancedWandbCallback(eval_freq=10000)
    
    # Evaluation callback
    eval_callback = EvaluationCallback(eval_env, eval_freq=10000, n_eval_episodes=3)
    
    callbacks = [img_callback, wandb_callback, eval_callback]
    
    print(f"✅ Callbacks initialized:")
    print(f"  - SafeImageRecorderCallback (every 20k steps)")
    print(f"  - EnhancedWandbCallback (comprehensive logging)")
    print(f"  - EvaluationCallback (every 100k steps)")
    
    # Start training
    print(f"\n🎯 Starting training for {TIMESTEPS:,} timesteps...")
    print("=" * 60)
    
    try:
        # Train the model with all callbacks
        sac_model.learn(
            total_timesteps=TIMESTEPS,
            progress_bar=True,
            callback=callbacks
        )
        
        # Save the final model
        model_path = f"sac_improved_crypto_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sac_model.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Comprehensive evaluation
    print("\n🔍 Running comprehensive evaluation...")
    
    try:
        # Test on multiple episodes
        eval_episodes = 10
        eval_rewards = []
        eval_net_worths = []
        eval_trade_counts = []
        
        for episode in range(eval_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_trades = 0
            
            while True:
                action, _states = sac_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if info.get('trade_executed', False):
                    episode_trades += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_net_worths.append(info['net_worth'])
            eval_trade_counts.append(episode_trades)
            
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Net Worth=${info['net_worth']:.2f}, Trades={episode_trades}")
        
        # Calculate evaluation statistics
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_net_worth = np.mean(eval_net_worths)
        max_net_worth = np.max(eval_net_worths)
        mean_trades = np.mean(eval_trade_counts)
        
        print(f"\n📈 Evaluation Results ({eval_episodes} episodes):")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean Net Worth: ${mean_net_worth:.2f}")
        print(f"Max Net Worth: ${max_net_worth:.2f}")
        print(f"Mean Trades per Episode: {mean_trades:.1f}")
        
        # Log final evaluation to WandB
        if wandb.run is not None:
            wandb.log({
                "final_mean_reward": mean_reward,
                "final_std_reward": std_reward,
                "final_mean_net_worth": mean_net_worth,
                "final_max_net_worth": max_net_worth,
                "final_mean_trades": mean_trades,
                "total_training_timesteps": TIMESTEPS
            })
        
        # Visualize final performance
        print("\n🎨 Generating final performance visualization...")
        eval_env.render(mode='human', agent_name='Final_SAC_Improved_Evaluation')
        
        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n✨ Key Improvements Implemented:")
        print("🔹 Multi-dimensional action space support (3D actions)")
        print("🔹 Comprehensive reward component tracking")
        print("🔹 Enhanced logging with detailed action analysis")
        print("🔹 Safe callback integration with error handling")
        print("🔹 Periodic evaluation with performance tracking")
        print("🔹 Market regime awareness and volatility adjustments")
        print("🔹 Symmetric fee and hold duration penalties")
        print("🔹 Position size and action change penalties")
        print("🔹 Weights & Biases integration for experiment tracking")
        
        print(f"\n📊 Final Performance Summary:")
        print(f"Training Timesteps: {TIMESTEPS:,}")
        print(f"Final Mean Reward: {mean_reward:.2f}")
        print(f"Final Mean Net Worth: ${mean_net_worth:.2f}")
        print(f"Best Net Worth Achieved: ${wandb_callback.best_net_worth:.2f}")
        print(f"Model Saved: {model_path}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    
    # Close WandB run
    if wandb.run is not None:
        wandb.finish()
    
    print("\n🏁 Training and evaluation completed!")


if __name__ == "__main__":
    import torch  # Import torch for the policy_kwargs
    main()