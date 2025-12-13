import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import os

# Load data
try:
    data = pd.read_csv('BTCUSDT_data.csv', nrows=100)
    print(f"Loaded data shape: {data.shape}")
except Exception as e:
    print(f"Failed to load data: {e}")
    # Create dummy data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000
    })
    print("Using dummy data.")

try:
    from enhanced_trading_env import EnhancedTradingEnv
    env = EnhancedTradingEnv(data)
    obs = env.reset()
    print("Environment reset successful.")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward:.6f}, done={done}")
        # Check if reward components exist
        for key in ['reward_base', 'reward_fee', 'reward_action_change', 'reward_trend', 'reward_holding', 'reward_inertia', 'reward_closer', 'reward_overtrade', 'reward_episode']:
            if key in info:
                print(f"  {key}: {info[key]:.6f}")
            else:
                print(f"  {key}: MISSING")
        if done:
            print("Episode ended.")
            break
except Exception as e:
    print(f"Error during environment test: {e}")
    import traceback
    traceback.print_exc()