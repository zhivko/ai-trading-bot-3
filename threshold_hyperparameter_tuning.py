"""
Threshold Hyperparameter Tuning for Trading Environment

This script provides a comprehensive approach to tuning the buy_threshold and sell_threshold
hyperparameters in the ContinuousTradingEnv environment.
"""

import numpy as np
import pandas as pd
import optuna
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import warnings
warnings.filterwarnings('ignore')

# Import your custom environment
from custom_trading_env import ContinuousTradingEnv

class ThresholdTuner:
    def __init__(self, data_path, n_trials=100, n_eval_episodes=5):
        """
        Initialize the threshold hyperparameter tuner
        
        Args:
            data_path: Path to your trading data CSV
            n_trials: Number of Optuna trials for optimization
            n_eval_episodes: Number of episodes to evaluate each configuration
        """
        self.data_path = data_path
        self.n_trials = n_trials
        self.n_eval_episodes = n_eval_episodes
        self.df = pd.read_csv(data_path)
        self.best_params = None
        self.best_score = -np.inf
        
    def objective(self, trial):
        """
        Optuna objective function to optimize threshold parameters
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Average reward over evaluation episodes (to be maximized)
        """
        # Suggest threshold ranges
        buy_threshold = trial.suggest_float('buy_threshold', 0.0, 1.0, step=0.01)
        sell_threshold = trial.suggest_float('sell_threshold', 0.0, 1.0, step=0.01)
        
        # Create environment with suggested thresholds
        env = ContinuousTradingEnv(
            df=self.df.copy(),
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            initial_balance=10000,
            window_size=20
        )
        env = Monitor(env)
        
        # Create and train agent
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            tensorboard_log=None
        )
        
        # Train for a limited number of steps to speed up tuning
        model.learn(total_timesteps=10000, log_interval=None)
        
        # Evaluate the trained agent
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        # Calculate average reward
        avg_reward = np.mean(episode_rewards)
        
        # Update best parameters if this is better
        if avg_reward > self.best_score:
            self.best_score = avg_reward
            self.best_params = {
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'avg_reward': avg_reward
            }
        
        # Add regularization penalty for overtrading
        # Penalize very low thresholds that might cause excessive trading
        threshold_penalty = -0.1 * (buy_threshold + sell_threshold)
        
        return avg_reward + threshold_penalty
    
    def tune_thresholds(self, study_name="threshold_tuning"):
        """
        Run the threshold optimization using Optuna
        
        Args:
            study_name: Name for the Optuna study
        """
        print("Starting threshold hyperparameter tuning...")
        print(f"Running {self.n_trials} trials...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
        
        print("\n=== OPTIMIZATION COMPLETE ===")
        print(f"Best average reward: {self.best_score:.4f}")
        print(f"Best parameters:")
        print(f"  buy_threshold: {self.best_params['buy_threshold']:.4f}")
        print(f"  sell_threshold: {self.best_params['sell_threshold']:.4f}")
        
        return self.best_params
    
    def evaluate_threshold_sensitivity(self, base_params=None):
        """
        Evaluate sensitivity of different threshold values
        
        Args:
            base_params: Base parameters to compare against
        """
        if base_params is None:
            base_params = {'buy_threshold': 0.1, 'sell_threshold': 0.1}
        
        # Test range of threshold values
        threshold_range = np.arange(0.05, 0.5, 0.05)
        results = []
        
        for buy_thresh in threshold_range:
            for sell_thresh in threshold_range:
                # Create environment
                env = ContinuousTradingEnv(
                    df=self.df.copy(),
                    buy_threshold=buy_thresh,
                    sell_threshold=sell_thresh,
                    initial_balance=10000,
                    window_size=20
                )
                
                # Simple random agent for baseline evaluation
                episode_rewards = []
                for _ in range(3):  # Fewer episodes for sensitivity analysis
                    obs, _ = env.reset()
                    episode_reward = 0
                    terminated = False
                    truncated = False
                    
                    while not (terminated or truncated):
                        # Random action for baseline
                        action = np.random.uniform(-1, 1, size=(1,))
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                    
                    episode_rewards.append(episode_reward)
                
                avg_reward = np.mean(episode_rewards)
                results.append({
                    'buy_threshold': buy_thresh,
                    'sell_threshold': sell_thresh,
                    'avg_reward': avg_reward
                })
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv('threshold_sensitivity_analysis.csv', index=False)
        
        print("Threshold sensitivity analysis saved to threshold_sensitivity_analysis.csv")
        return results_df
    
    def create_threshold_heatmap(self, results_df):
        """
        Create a heatmap visualization of threshold performance
        
        Args:
            results_df: DataFrame with threshold sensitivity results
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Pivot data for heatmap
        heatmap_data = results_df.pivot(
            index='buy_threshold', 
            columns='sell_threshold', 
            values='avg_reward'
        )
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='viridis')
        plt.title('Threshold Performance Heatmap')
        plt.xlabel('Sell Threshold')
        plt.ylabel('Buy Threshold')
        plt.tight_layout()
        plt.savefig('threshold_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Threshold heatmap saved to threshold_heatmap.png")

def run_comprehensive_tuning(data_path):
    """
    Run comprehensive threshold tuning analysis
    
    Args:
        data_path: Path to your trading data
    """
    tuner = ThresholdTuner(data_path, n_trials=50)  # Reduced for faster execution
    
    # 1. Run optimization
    best_params = tuner.tune_thresholds()
    
    # 2. Analyze threshold sensitivity
    print("\n=== THRESHOLD SENSITIVITY ANALYSIS ===")
    sensitivity_results = tuner.evaluate_threshold_sensitivity()
    
    # 3. Create visualization
    tuner.create_threshold_heatmap(sensitivity_results)
    
    # 4. Generate recommendations
    print("\n=== THRESHOLD TUNING RECOMMENDATIONS ===")
    print(f"1. Optimal buy_threshold: {best_params['buy_threshold']:.4f}")
    print(f"2. Optimal sell_threshold: {best_params['sell_threshold']:.4f}")
    print(f"3. Expected performance: {best_params['avg_reward']:.4f} average reward")
    
    # Additional insights
    low_threshold_mask = sensitivity_results['buy_threshold'] <= 0.1
    high_threshold_mask = sensitivity_results['buy_threshold'] >= 0.3
    
    low_performance = sensitivity_results[low_threshold_mask]['avg_reward'].mean()
    high_performance = sensitivity_results[high_threshold_mask]['avg_reward'].mean()
    
    print(f"4. Low threshold (≤0.1) average performance: {low_performance:.4f}")
    print(f"5. High threshold (≥0.3) average performance: {high_performance:.4f}")
    
    return best_params

if __name__ == "__main__":
    # Example usage
    data_path = "BTCUSDT_data.csv"  # Update with your data path
    
    print("=== THRESHOLD HYPERPARAMETER TUNING ===")
    print("This script will optimize your buy_threshold and sell_threshold parameters")
    print("for the ContinuousTradingEnv environment.\n")
    
    # Check if data file exists
    import os
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found!")
        print("Please update the data_path variable with the correct file path.")
        exit(1)
    
    # Run comprehensive tuning
    optimal_params = run_comprehensive_tuning(data_path)
    
    print("\n=== FINAL RECOMMENDATIONS ===")
    print("Use these optimized parameters in your environment:")
    print(f"ContinuousTradingEnv(df, buy_threshold={optimal_params['buy_threshold']:.4f}, "
          f"sell_threshold={optimal_params['sell_threshold']:.4f})")