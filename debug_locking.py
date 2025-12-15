#!/usr/bin/env python3
"""
Debug script to understand the locking mechanism behavior.
"""

import pandas as pd
import numpy as np
from improved_trading_env import ImprovedTradingEnv

def create_test_data():
    """Create test data with predictable price movements."""
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # Create a simple upward trend with some noise
    base_price = 100
    prices = []
    current_price = base_price
    
    for i in range(100):
        # Upward trend with noise
        noise = np.random.normal(0, 0.5)
        current_price = current_price * (1 + 0.001 + noise / 100)
        prices.append(current_price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(-200, 200) for _ in prices]
    })
    
    return df

def debug_locking():
    """Debug the locking mechanism step by step."""
    print("🔍 Debugging Locking Mechanism")
    print("=" * 40)
    
    # Create test data
    test_df = create_test_data()
    
    # Create environment
    env = ImprovedTradingEnv(
        df=test_df,
        initial_balance=10000,
        window_size=20,
        max_hold_steps=5
    )
    
    print("=== Test 1: No Locking (duration = 0) ===")
    obs, info = env.reset()
    print(f"Initial state - Shares held: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")
    
    # First action: Buy with no lock
    action1 = [0.5, 1.0, 0.0]  # Buy, full size, no lock
    print(f"Action 1: {action1}")
    obs, reward, terminated, truncated, info = env.step(action1)
    print(f"After Action 1 - Trade executed: {info['trade_executed']}, Shares: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")
    
    # Second action: Sell with no lock
    action2 = [-0.5, 1.0, 0.0]  # Sell, full size, no lock
    print(f"Action 2: {action2}")
    obs, reward, terminated, truncated, info = env.step(action2)
    print(f"After Action 2 - Trade executed: {info['trade_executed']}, Shares: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")
    
    print("\n=== Test 2: With Locking (duration = 1.0) ===")
    obs, info = env.reset()
    print(f"Initial state - Shares held: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")
    
    # First action: Buy with maximum lock
    action1 = [0.5, 1.0, 1.0]  # Buy, full size, max lock
    print(f"Action 1: {action1}")
    obs, reward, terminated, truncated, info = env.step(action1)
    print(f"After Action 1 - Trade executed: {info['trade_executed']}, Shares: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")
    
    # Second action: Try to sell (should be ignored due to lock)
    action2 = [-1.0, 1.0, 0.0]  # Try to sell everything
    print(f"Action 2: {action2}")
    obs, reward, terminated, truncated, info = env.step(action2)
    print(f"After Action 2 - Trade executed: {info['trade_executed']}, Shares: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")
    
    # Third action: Still locked
    action3 = [0.0, 0.0, 0.0]  # Hold
    print(f"Action 3: {action3}")
    obs, reward, terminated, truncated, info = env.step(action3)
    print(f"After Action 3 - Trade executed: {info['trade_executed']}, Shares: {info['shares_held']}, Lock: {info.get('lock_remaining', 0)}")

if __name__ == "__main__":
    debug_locking()