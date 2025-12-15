"""
Quick Threshold Tuning for Trading Environment

Simple script to quickly test and tune threshold parameters for your trading bot.
Run this script to get immediate insights into optimal threshold values.
"""

import numpy as np
import pandas as pd
from custom_trading_env import ContinuousTradingEnv
import matplotlib.pyplot as plt
import seaborn as sns

def quick_threshold_test(data_path, n_episodes=3):
    """
    Quick test of different threshold combinations
    
    Args:
        data_path: Path to your trading data CSV
        n_episodes: Number of episodes per threshold combination
    """
    print("=== QUICK THRESHOLD TESTING ===")
    print("Testing different threshold combinations...\n")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Define threshold ranges to test
    threshold_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    results = []
    
    for buy_thresh in threshold_values:
        for sell_thresh in threshold_values:
            print(f"Testing: buy_threshold={buy_thresh}, sell_threshold={sell_thresh}")
            
            # Create environment
            env = ContinuousTradingEnv(
                df=df.copy(),
                buy_threshold=buy_thresh,
                sell_threshold=sell_thresh,
                initial_balance=10000,
                window_size=20
            )
            
            # Run episodes with random actions for baseline
            episode_rewards = []
            episode_profits = []
            trade_counts = []
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_profit = 0
                trades = 0
                terminated = False
                truncated = False
                
                while not (terminated or truncated):
                    # Use a simple strategy: buy when action > buy_thresh, sell when action < -sell_thresh
                    # For testing, use random actions
                    action = np.random.uniform(-1, 1, size=(1,))
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # Track trades
                    if info.get('trade_executed', False):
                        trades += 1
                
                episode_rewards.append(episode_reward)
                episode_profit = env.net_worth - env.initial_balance
                episode_profits.append(episode_profit)
                trade_counts.append(trades)
            
            # Calculate averages
            avg_reward = np.mean(episode_rewards)
            avg_profit = np.mean(episode_profits)
            avg_trades = np.mean(trade_counts)
            
            results.append({
                'buy_threshold': buy_thresh,
                'sell_threshold': sell_thresh,
                'avg_reward': avg_reward,
                'avg_profit': avg_profit,
                'avg_trades': avg_trades,
                'profit_per_trade': avg_profit / max(avg_trades, 1)  # Avoid division by zero
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best combinations
    best_reward_idx = results_df['avg_reward'].idxmax()
    best_profit_idx = results_df['avg_profit'].idxmax()
    best_efficiency_idx = results_df['profit_per_trade'].idxmax()
    
    print("\n=== RESULTS SUMMARY ===")
    print(f"Best for Reward: buy_threshold={results_df.loc[best_reward_idx, 'buy_threshold']:.2f}, "
          f"sell_threshold={results_df.loc[best_reward_idx, 'sell_threshold']:.2f}, "
          f"reward={results_df.loc[best_reward_idx, 'avg_reward']:.4f}")
    
    print(f"Best for Profit: buy_threshold={results_df.loc[best_profit_idx, 'buy_threshold']:.2f}, "
          f"sell_threshold={results_df.loc[best_profit_idx, 'sell_threshold']:.2f}, "
          f"profit=${results_df.loc[best_profit_idx, 'avg_profit']:.2f}")
    
    print(f"Best for Efficiency: buy_threshold={results_df.loc[best_efficiency_idx, 'buy_threshold']:.2f}, "
          f"sell_threshold={results_df.loc[best_efficiency_idx, 'sell_threshold']:.2f}, "
          f"profit/trade=${results_df.loc[best_efficiency_idx, 'profit_per_trade']:.2f}")
    
    # Save results
    results_df.to_csv('quick_threshold_results.csv', index=False)
    print(f"\nDetailed results saved to 'quick_threshold_results.csv'")
    
    return results_df

def plot_threshold_analysis(results_df):
    """
    Create visualization of threshold analysis results
    """
    print("\nCreating threshold analysis plots...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward heatmap
    reward_pivot = results_df.pivot(index='buy_threshold', columns='sell_threshold', values='avg_reward')
    sns.heatmap(reward_pivot, annot=True, fmt='.4f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Average Reward by Threshold Combination')
    
    # 2. Profit heatmap
    profit_pivot = results_df.pivot(index='buy_threshold', columns='sell_threshold', values='avg_profit')
    sns.heatmap(profit_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,1])
    axes[0,1].set_title('Average Profit by Threshold Combination')
    
    # 3. Trade count heatmap
    trades_pivot = results_df.pivot(index='buy_threshold', columns='sell_threshold', values='avg_trades')
    sns.heatmap(trades_pivot, annot=True, fmt='.1f', cmap='Blues', ax=axes[1,0])
    axes[1,0].set_title('Average Trade Count by Threshold Combination')
    
    # 4. Efficiency (profit per trade) heatmap
    efficiency_pivot = results_df.pivot(index='buy_threshold', columns='sell_threshold', values='profit_per_trade')
    sns.heatmap(efficiency_pivot, annot=True, fmt='.2f', cmap='plasma', ax=axes[1,1])
    axes[1,1].set_title('Profit per Trade by Threshold Combination')
    
    plt.tight_layout()
    plt.savefig('threshold_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis plots saved to 'threshold_analysis_plots.png'")

def threshold_recommendations(results_df):
    """
    Provide specific recommendations based on results
    """
    print("\n=== THRESHOLD TUNING RECOMMENDATIONS ===")
    
    # Analyze trade frequency impact
    high_trade_mask = results_df['avg_trades'] > results_df['avg_trades'].median()
    low_trade_performance = results_df[high_trade_mask]['avg_profit'].mean()
    high_trade_performance = results_df[~high_trade_mask]['avg_profit'].mean()
    
    print(f"1. Trade Frequency Analysis:")
    print(f"   - High frequency trading (>{results_df['avg_trades'].median():.1f} trades): ${low_trade_performance:.2f} avg profit")
    print(f"   - Low frequency trading (≤{results_df['avg_trades'].median():.1f} trades): ${high_trade_performance:.2f} avg profit")
    
    # Find optimal ranges
    best_10_percent = results_df.nlargest(int(len(results_df) * 0.1), 'avg_profit')
    
    print(f"\n2. Optimal Range Analysis:")
    print(f"   - Best buy_threshold range: {best_10_percent['buy_threshold'].min():.2f} - {best_10_percent['buy_threshold'].max():.2f}")
    print(f"   - Best sell_threshold range: {best_10_percent['sell_threshold'].min():.2f} - {best_10_percent['sell_threshold'].max():.2f}")
    
    # Specific recommendations
    best_overall = results_df.loc[results_df['avg_profit'].idxmax()]
    
    print(f"\n3. Final Recommendations:")
    print(f"   - Primary recommendation: buy_threshold={best_overall['buy_threshold']:.2f}, sell_threshold={best_overall['sell_threshold']:.2f}")
    print(f"   - Expected profit: ${best_overall['avg_profit']:.2f}")
    print(f"   - Expected trades per episode: {best_overall['avg_trades']:.1f}")
    
    # Conservative vs aggressive strategies
    conservative = results_df[(results_df['buy_threshold'] >= 0.2) & (results_df['sell_threshold'] >= 0.2)]
    aggressive = results_df[(results_df['buy_threshold'] <= 0.1) & (results_df['sell_threshold'] <= 0.1)]
    
    if not conservative.empty and not aggressive.empty:
        cons_perf = conservative['avg_profit'].mean()
        agg_perf = aggressive['avg_profit'].mean()
        
        print(f"\n4. Strategy Comparison:")
        print(f"   - Conservative (thresholds ≥ 0.2): ${cons_perf:.2f} avg profit")
        print(f"   - Aggressive (thresholds ≤ 0.1): ${agg_perf:.2f} avg profit")
        
        if cons_perf > agg_perf:
            print(f"   - Recommendation: Use conservative approach for better risk-adjusted returns")
        else:
            print(f"   - Recommendation: Use aggressive approach for higher returns")

def run_quick_tuning(data_path="BTCUSDT_data.csv"):
    """
    Run the complete quick threshold tuning analysis
    """
    print("=== QUICK THRESHOLD TUNING ANALYSIS ===")
    print(f"Using data from: {data_path}\n")
    
    try:
        # Run quick test
        results = quick_threshold_test(data_path, n_episodes=2)  # Reduced episodes for speed
        
        # Create visualizations
        plot_threshold_analysis(results)
        
        # Generate recommendations
        threshold_recommendations(results)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print("Files created:")
        print("- quick_threshold_results.csv (detailed data)")
        print("- threshold_analysis_plots.png (visualization)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your data file exists and has the correct format.")

if __name__ == "__main__":
    # Run quick analysis
    run_quick_tuning()