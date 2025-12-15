# Threshold Hyperparameter Tuning Guide

## Overview

This guide explains how to tune the `buy_threshold` and `sell_threshold` hyperparameters in your ContinuousTradingEnv to optimize trading performance.

## Understanding Thresholds

### What are thresholds?

In your trading environment, thresholds control when trades are executed:

- **buy_threshold**: Minimum action value required to execute a buy trade
- **sell_threshold**: Minimum absolute action value required to execute a sell trade

### How they work:

```python
# Buy condition: action > buy_threshold
if current_action > self.buy_threshold:
    # Execute buy trade

# Sell condition: action < -sell_threshold  
if current_action < -self.sell_threshold:
    # Execute sell trade
```

### Impact on trading:

- **Low thresholds** (0.0-0.1): More frequent trading, higher risk
- **High thresholds** (0.3+): Conservative trading, fewer trades
- **Balanced thresholds** (0.1-0.2): Moderate trading frequency

## Tuning Strategies

### 1. Quick Analysis (Recommended Start)

Use `quick_threshold_tuning.py` for immediate insights:

```bash
python quick_threshold_tuning.py
```

This script:
- Tests 7 different threshold values (0.0 to 0.3)
- Evaluates 49 combinations
- Provides immediate recommendations
- Creates visualization plots

### 2. Comprehensive Optimization

Use `threshold_hyperparameter_tuning.py` for detailed optimization:

```bash
python threshold_hyperparameter_tuning.py
```

This script:
- Uses Optuna for Bayesian optimization
- Trains actual RL agents for each configuration
- Provides statistical significance testing
- Takes longer but gives optimal results

## Step-by-Step Tuning Process

### Step 1: Quick Analysis

```python
from quick_threshold_tuning import run_quick_tuning

# Run quick analysis
results = run_quick_tuning("BTCUSDT_data.csv")
```

**Expected output:**
- Best threshold combinations for reward, profit, and efficiency
- Analysis plots saved as PNG files
- CSV file with detailed results

### Step 2: Analyze Results

Key metrics to examine:

1. **Average Reward**: Overall performance metric
2. **Average Profit**: Dollar-based performance
3. **Trade Frequency**: Number of trades per episode
4. **Profit per Trade**: Efficiency metric

### Step 3: Apply Optimal Thresholds

Based on results, update your environment:

```python
# Use the best performing thresholds
env = ContinuousTradingEnv(
    df=data,
    buy_threshold=0.15,  # Example optimal value
    sell_threshold=0.20,  # Example optimal value
    initial_balance=10000,
    window_size=20
)
```

## Threshold Tuning Best Practices

### 1. Start with Quick Analysis

Always begin with the quick analysis to get baseline insights:

```python
# Quick test first
quick_results = quick_threshold_test(data_path, n_episodes=3)
```

### 2. Consider Your Trading Strategy

**For Scalping (High Frequency):**
- Lower thresholds: 0.05 - 0.15
- More trades, smaller profits per trade
- Higher risk but potentially higher returns

**For Swing Trading (Lower Frequency):**
- Higher thresholds: 0.20 - 0.40
- Fewer trades, larger profits per trade
- Lower risk, more conservative approach

**For Balanced Approach:**
- Moderate thresholds: 0.10 - 0.25
- Balanced trade frequency
- Risk-adjusted returns

### 3. Validate Across Time Periods

Test your optimal thresholds on different market conditions:

```python
# Test on bull market data
bull_market_results = test_thresholds_on_period(bull_data, optimal_thresholds)

# Test on bear market data  
bear_market_results = test_thresholds_on_period(bear_data, optimal_thresholds)

# Test on sideways market data
sideways_results = test_thresholds_on_period(sideways_data, optimal_thresholds)
```

### 4. Monitor Performance Over Time

Regularly re-tune thresholds as market conditions change:

```python
# Monthly re-tuning
monthly_tune_thresholds(recent_data)
```

## Common Threshold Patterns

### Pattern 1: Symmetric Thresholds
```python
buy_threshold = 0.15
sell_threshold = 0.15
```
**Best for:** Balanced strategies, equal buy/sell sensitivity

### Pattern 2: Asymmetric Thresholds
```python
buy_threshold = 0.10  # More sensitive to buy signals
sell_threshold = 0.20  # Less sensitive to sell signals
```
**Best for:** Bull markets, momentum strategies

### Pattern 3: Conservative Approach
```python
buy_threshold = 0.25
sell_threshold = 0.25
```
**Best for:** Risk-averse strategies, volatile markets

### Pattern 4: Aggressive Approach
```python
buy_threshold = 0.05
sell_threshold = 0.05
```
**Best for:** High-confidence signals, trending markets

## Troubleshooting

### Issue 1: No Trades Executed

**Symptoms:** Trade count = 0 for all episodes
**Solution:** Lower thresholds or check action space bounds

```python
# Lower thresholds
env = ContinuousTradingEnv(df, buy_threshold=0.01, sell_threshold=0.01)
```

### Issue 2: Too Many Trades

**Symptoms:** Very high trade frequency, poor performance
**Solution:** Increase thresholds to filter weak signals

```python
# Higher thresholds
env = ContinuousTradingEnv(df, buy_threshold=0.20, sell_threshold=0.20)
```

### Issue 3: Inconsistent Results

**Symptoms:** High variance in performance across episodes
**Solution:** Increase episode count for more stable evaluation

```python
# More episodes for stable evaluation
results = quick_threshold_test(data_path, n_episodes=10)
```

### Issue 4: Poor Performance

**Symptoms:** Negative profits across all threshold combinations
**Solution:** Check data quality and environment setup

```python
# Verify data format
print(data.head())
print(data.columns)

# Check environment parameters
env = ContinuousTradingEnv(
    df=data,
    initial_balance=10000,
    commission=0.001,  # Ensure reasonable commission
    window_size=20
)
```

## Integration with RL Training

### Use Optimal Thresholds in Training

```python
from stable_baselines3 import PPO

# Create environment with tuned thresholds
env = ContinuousTradingEnv(
    df=training_data,
    buy_threshold=0.15,  # Your optimal values
    sell_threshold=0.20,
    initial_balance=10000,
    window_size=20
)

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Threshold as Trainable Parameter

For advanced use cases, make thresholds part of the action space:

```python
class AdaptiveTradingEnv(ContinuousTradingEnv):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Add thresholds to action space as additional parameters
        self.action_space = spaces.Dict({
            'trade_action': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'buy_threshold': spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32),
            'sell_threshold': spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32)
        })
```

## Performance Monitoring

### Track Threshold Performance

```python
class ThresholdMonitor:
    def __init__(self):
        self.performance_history = []
    
    def evaluate_thresholds(self, buy_threshold, sell_threshold, n_episodes=10):
        env = ContinuousTradingEnv(data, buy_threshold, sell_threshold)
        
        results = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            # ... run episode
            
            results.append({
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'reward': episode_reward,
                'profit': env.net_worth - env.initial_balance
            })
        
        return pd.DataFrame(results)
```

## Conclusion

Threshold tuning is crucial for optimizing your trading bot's performance. Start with the quick analysis to get immediate insights, then use the comprehensive optimization for fine-tuning. Remember to regularly re-evaluate thresholds as market conditions change.

**Key takeaways:**
1. Start with quick analysis for baseline insights
2. Consider your trading strategy and risk tolerance
3. Test across different market conditions
4. Monitor and adjust thresholds regularly
5. Use optimal thresholds in your RL training pipeline