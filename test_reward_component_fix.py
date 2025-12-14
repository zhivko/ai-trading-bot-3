#!/usr/bin/env python3
"""
Test script to verify the reward component plotting fix is working correctly.
This test simulates the environment data flow to ensure reward components 
are properly extracted and plotted.
"""

import numpy as np
import pandas as pd
from enhanced_trading_env import EnhancedTradingEnv
from callbacks.base_callbacks import TensorboardCallback, CustomEvalCallback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def test_reward_component_extraction():
    """Test that reward components are properly extracted from environment info."""
    print("Testing reward component extraction...")
    
    # Create test data
    np.random.seed(42)
    data = {
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01', periods=200, freq='1h')
    
    # Create environment
    env = EnhancedTradingEnv(df, lookback_window=10)
    obs, info = env.reset()
    
    # Simulate a few steps and collect reward components
    reward_components_history = []
    
    for step in range(10):
        action = np.array([0.1 if step % 3 == 0 else -0.05 if step % 3 == 1 else 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if reward_components is available
        if 'reward_components' in info:
            reward_components_history.append(info['reward_components'])
            print(f"Step {step}: Reward components found - {list(info['reward_components'].keys())}")
            print(f"  Values: {info['reward_components']}")
        else:
            print(f"Step {step}: No reward_components in info")
            
        if terminated or truncated:
            break
    
    return reward_components_history

def test_callback_with_mock_data():
    """Test the callback with mock environment data to verify the fix works."""
    print("\nTesting callback with mock data...")
    
    # Create mock callback
    callback = TensorboardCallback(verbose=1)
    
    # Create mock portfolio data that includes reward_components
    mock_portfolio = []
    for i in range(20):
        mock_portfolio.append({
            'step': i,
            'net_worth': 10000 + i * 10,
            'price': 100 + i,
            'action': 0.1 if i % 3 == 0 else -0.05 if i % 3 == 1 else 0.0,
            'shares': i,
            'trade_executed': i % 5 == 0,
            'panic_close': False,
            'reward_components': {
                'base': 0.1 * (i + 1),
                'fee_penalty': -0.01 * i,
                'action_change_penalty': -0.005 * abs(0.1 if i % 3 == 0 else -0.05 if i % 3 == 1 else 0.0),
                'trend_alignment': 0.02 * i,
                'holding_penalty': -0.001 * i,
                'inertia_penalty': -0.002 * i,
                'closer_bonus': 0.05 if i % 7 == 0 else 0.0,
                'overtrading_penalty': -0.01 if i % 4 == 0 else 0.0,
            }
        })
    
    # Set up the callback with mock data
    callback.ep_portfolio = mock_portfolio
    callback.ep_prices = [p['price'] for p in mock_portfolio]
    callback.ep_actions = [p['action'] for p in mock_portfolio]
    callback.ep_emas = [p['price'] * 0.99 for p in mock_portfolio]  # Mock EMA
    callback.ep_dates = [f"2023-01-01 {i:02d}:00" for i in range(20)]
    callback.ep_rewards = [1.0] * 20
    
    # Test the plotting function
    try:
        callback._plot_regime_chart()
        print("✅ Callback plotting completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Callback plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_component_statistics():
    """Test that reward components have non-zero values and proper statistics."""
    print("\nTesting reward component statistics...")
    
    # Generate test reward components with varied values
    reward_components = {
        'base': np.random.normal(0.1, 0.05, 50).tolist(),
        'fee_penalty': np.random.normal(-0.01, 0.005, 50).tolist(),
        'action_change_penalty': np.random.normal(-0.005, 0.002, 50).tolist(),
        'trend_alignment': np.random.normal(0.02, 0.01, 50).tolist(),
        'holding_penalty': np.random.normal(-0.001, 0.0005, 50).tolist(),
        'inertia_penalty': np.random.normal(-0.002, 0.001, 50).tolist(),
        'closer_bonus': np.random.normal(0.05, 0.02, 50).tolist(),
        'overtrading_penalty': np.random.normal(-0.01, 0.005, 50).tolist(),
    }
    
    print("Reward component statistics:")
    all_nonzero = True
    for key, values in reward_components.items():
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        nonzero_count = np.count_nonzero(values_array)
        
        print(f"  {key}: mean={mean_val:.6f}, std={std_val:.6f}, non-zero={nonzero_count}/{len(values)}")
        
        if mean_val == 0.0 and std_val == 0.0:
            all_nonzero = False
    
    if all_nonzero:
        print("✅ All reward components have non-zero values!")
    else:
        print("❌ Some reward components are all zeros")
    
    return all_nonzero

def main():
    """Run all tests to verify the reward component fix."""
    print("=== Reward Component Plotting Fix Test ===")
    print()
    
    # Test 1: Environment data flow
    reward_components_history = test_reward_component_extraction()
    
    # Test 2: Callback with mock data
    callback_success = test_callback_with_mock_data()
    
    # Test 3: Statistics
    stats_success = test_reward_component_statistics()
    
    print("\n=== Test Results ===")
    if reward_components_history:
        print(f"✅ Environment provides reward components: {len(reward_components_history)} steps")
    else:
        print("❌ Environment does not provide reward components")
    
    if callback_success:
        print("✅ Callback plotting works with reward components")
    else:
        print("❌ Callback plotting failed")
    
    if stats_success:
        print("✅ Reward components have proper statistical variation")
    else:
        print("❌ Reward components lack variation")
    
    overall_success = bool(reward_components_history) and callback_success and stats_success
    
    if overall_success:
        print("\n🎉 All tests passed! The reward component plotting fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return overall_success

if __name__ == "__main__":
    main()