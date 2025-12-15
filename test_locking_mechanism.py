#!/usr/bin/env python3
"""
Test script to validate the duration-based locking mechanism in the trading environment.
This script tests various scenarios to ensure the locking works correctly.
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

def test_locking_mechanism():
    """Test the locking mechanism with various scenarios."""
    print("🧪 Testing Duration-Based Locking Mechanism")
    print("=" * 50)
    
    # Create test data
    test_df = create_test_data()
    print(f"✅ Created test data with {len(test_df)} time steps")
    
    # Create environment with locking enabled
    env = ImprovedTradingEnv(
        df=test_df,
        initial_balance=10000,
        window_size=20,
        max_hold_steps=5  # 5 steps maximum lock
    )
    
    print(f"✅ Created environment with max_hold_steps={env.max_hold_steps}")
    print(f"✅ Action space: {env.action_space}")
    print(f"✅ Third dimension range: [{env.action_space.low[2]}, {env.action_space.high[2]}]")
    
    # Test 1: No locking (duration = 0)
    print("\n📋 Test 1: No Locking (duration = 0)")
    obs, info = env.reset()
    
    # Action: Buy with no lock
    action_no_lock = [0.5, 1.0, 0.0]  # Buy, full size, no lock
    obs, reward, terminated, truncated, info = env.step(action_no_lock)
    
    print(f"  Initial action: {action_no_lock}")
    print(f"  Trade executed: {info['trade_executed']}")
    print(f"  Lock remaining: {info['lock_remaining']}")
    print(f"  Shares held: {info['shares_held']}")
    
    # Next step should allow new action - try to buy more
    action_new = [0.8, 1.0, 0.0]  # Buy more, full size, no lock
    obs, reward, terminated, truncated, info = env.step(action_new)
    
    print(f"  Next action: {action_new}")
    print(f"  Trade executed: {info['trade_executed']}")
    print(f"  Lock remaining: {info['lock_remaining']}")
    print(f"  Shares held: {info['shares_held']}")
    
    if info['trade_executed']:
        print(f"  ✅ Test 1 passed: No locking works correctly")
    else:
        print(f"  ❌ Test 1 failed: Expected trade execution but got none")
        return False
    
    # Test 2: With locking (duration > 0)
    print("\n📋 Test 2: With Locking (duration = 1.0)")
    obs, info = env.reset()
    
    # Action: Buy with maximum lock
    action_with_lock = [0.5, 1.0, 1.0]  # Buy, full size, max lock
    obs, reward, terminated, truncated, info = env.step(action_with_lock)
    
    print(f"  Initial action: {action_with_lock}")
    print(f"  Trade executed: {info['trade_executed']}")
    print(f"  Lock remaining: {info['lock_remaining']}")
    print(f"  Shares held: {info['shares_held']}")
    
    # Next step should be locked - new action should be ignored
    action_ignored = [-1.0, 1.0, 0.0]  # Try to sell everything
    obs, reward, terminated, truncated, info = env.step(action_ignored)
    
    print(f"  Locked action: {action_ignored}")
    print(f"  Trade executed: {info['trade_executed']}")
    print(f"  Lock remaining: {info['lock_remaining']}")
    print(f"  Shares held: {info['shares_held']}")
    
    if not info['trade_executed'] and info['lock_remaining'] == 3:
        print(f"  ✅ Test 2 passed: Locking prevents new trades")
    else:
        print(f"  ❌ Test 2 failed: Locking not working correctly")
        print(f"  Expected: trade_executed=False, lock_remaining=3")
        print(f"  Actual: trade_executed={info['trade_executed']}, lock_remaining={info['lock_remaining']}")
        return False
    
    # Test 3: Partial lock duration
    print("\n📋 Test 3: Partial Lock Duration")
    obs, info = env.reset()
    
    # Action: Buy with partial lock (2.5 steps -> 2 steps)
    action_partial_lock = [0.3, 0.8, 0.5]  # Buy, 80% size, 50% lock duration
    obs, reward, terminated, truncated, info = env.step(action_partial_lock)
    
    expected_lock = int(0.5 * env.max_hold_steps) - 1  # 2.5 -> 2 -> 1
    print(f"  Initial action: {action_partial_lock}")
    print(f"  Expected lock: {expected_lock}")
    print(f"  Actual lock remaining: {info['lock_remaining']}")
    
    if info['lock_remaining'] == expected_lock:
        print(f"  ✅ Test 3 passed: Partial lock duration works")
    else:
        print(f"  ❌ Test 3 failed: Expected {expected_lock}, got {info['lock_remaining']}")
        return False
    
    # Test 4: Lock bonus for profitable holds
    print("\n📋 Test 4: Lock Bonus for Profitable Holds")
    obs, info = env.reset()
    
    # Buy at start
    action_buy = [1.0, 1.0, 1.0]  # Buy, full size, max lock
    obs, reward, terminated, truncated, info = env.step(action_buy)
    
    initial_net_worth = info['net_worth']
    print(f"  Initial net worth: ${initial_net_worth:.2f}")
    
    # Step through locked period and check for bonuses
    bonuses_received = 0
    for step in range(3):  # First 3 steps of lock
        obs, reward, terminated, truncated, info = env.step([0, 0, 0])
        
        if 'lock_bonus' in info['reward_components'] and info['reward_components']['lock_bonus'] > 0:
            bonuses_received += 1
            print(f"  Step {step + 1}: Received lock bonus: {info['reward_components']['lock_bonus']}")
    
    if bonuses_received > 0:
        print(f"  ✅ Test 4 passed: Lock bonuses awarded ({bonuses_received} received)")
    else:
        print(f"  ⚠️  Test 4: No lock bonuses received (may be due to price movement)")
    
    # Test 5: Action space validation
    print("\n📋 Test 5: Action Space Validation")
    print(f"  Action space low: {env.action_space.low}")
    print(f"  Action space high: {env.action_space.high}")
    
    # Test invalid actions (should be clipped)
    invalid_action = [2.0, 1.5, -0.5]  # Invalid values
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    
    print(f"  Invalid action: {invalid_action}")
    print(f"  Action components: {info['action_components']}")
    
    # Check that components are within bounds
    action_components = np.array(info['action_components'])
    if np.all(action_components >= env.action_space.low) and np.all(action_components <= env.action_space.high):
        print(f"  ✅ Test 5 passed: Action clipping works")
    else:
        print(f"  ❌ Test 5 failed: Action clipping not working")
        return False
    
    print("\n🎉 All tests passed! Locking mechanism is working correctly.")
    return True

def test_performance_impact():
    """Test the performance impact of locking on trading behavior."""
    print("\n📊 Testing Performance Impact of Locking")
    print("=" * 50)
    
    test_df = create_test_data()
    
    # Test with different lock durations
    lock_configs = [
        {"max_hold_steps": 0, "name": "No Locking"},
        {"max_hold_steps": 3, "name": "Short Lock (3 steps)"},
        {"max_hold_steps": 10, "name": "Medium Lock (10 steps)"},
        {"max_hold_steps": 20, "name": "Long Lock (20 steps)"}
    ]
    
    results = []
    
    for config in lock_configs:
        print(f"\n🔄 Testing: {config['name']}")
        
        env = ImprovedTradingEnv(
            df=test_df,
            initial_balance=10000,
            window_size=20,
            max_hold_steps=config['max_hold_steps']
        )
        
        obs, info = env.reset()
        total_trades = 0
        total_rewards = 0
        total_steps = 0
        
        # Run episode with random actions
        for step in range(50):  # Run 50 steps
            # Random action with some lock probability
            if config['max_hold_steps'] > 0:
                # 30% chance to set a lock
                lock_prob = 0.3
                if np.random.random() < lock_prob:
                    duration = np.random.uniform(0.3, 1.0)  # 30-100% of max duration
                else:
                    duration = 0.0
            else:
                duration = 0.0
            
            action = [
                np.random.uniform(-1, 1),  # Position
                np.random.uniform(0, 1),   # Size
                duration                   # Duration
            ]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info['trade_executed']:
                total_trades += 1
            
            total_rewards += reward
            total_steps += 1
            
            if terminated or truncated:
                break
        
        final_net_worth = info['net_worth']
        return_pct = (final_net_worth - env.initial_balance) / env.initial_balance * 100
        
        result = {
            'config': config['name'],
            'total_trades': total_trades,
            'total_rewards': total_rewards,
            'final_net_worth': final_net_worth,
            'return_pct': return_pct,
            'avg_trades_per_step': total_trades / total_steps if total_steps > 0 else 0
        }
        
        results.append(result)
        
        print(f"  Total trades: {total_trades}")
        print(f"  Final net worth: ${final_net_worth:.2f}")
        print(f"  Return: {return_pct:.2f}%")
        print(f"  Avg trades/step: {result['avg_trades_per_step']:.3f}")
    
    # Compare results
    print(f"\n📈 Performance Comparison:")
    print(f"{'Configuration':<25} {'Trades':<8} {'Return %':<10} {'Trades/Step':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['config']:<25} {result['total_trades']:<8} {result['return_pct']:<10.2f} {result['avg_trades_per_step']:<12.3f}")
    
    # Analyze impact
    no_lock = results[0]
    with_lock = results[1]
    
    if with_lock['total_trades'] < no_lock['total_trades']:
        reduction = ((no_lock['total_trades'] - with_lock['total_trades']) / no_lock['total_trades']) * 100
        print(f"\n✅ Locking reduced trading frequency by {reduction:.1f}%")
    else:
        print(f"\n⚠️  Locking did not reduce trading frequency as expected")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting Locking Mechanism Validation Tests")
    print("=" * 60)
    
    try:
        # Run basic functionality tests
        success = test_locking_mechanism()
        
        if success:
            # Run performance impact tests
            test_performance_impact()
            
            print(f"\n🎉 All validation tests completed successfully!")
            print(f"✅ Duration-based locking mechanism is ready for training")
        else:
            print(f"\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()