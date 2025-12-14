#!/usr/bin/env python3
"""
Test script to verify the fixes applied to custom_trading_env.py:
- Fix A: __init__ method initializes essential attributes before calling _process_data
- Fix B: _process_data uses .copy() to avoid SettingWithCopyWarning
"""

import pandas as pd
import numpy as np
import warnings
from custom_trading_env import ContinuousTradingEnv

def test_fixes():
    """Test both fixes in custom_trading_env.py"""
    
    print("=" * 60)
    print("Testing fixes applied to custom_trading_env.py")
    print("=" * 60)
    
    # Create test data with required columns including timestamp
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=200, freq='1h'),
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200),
        'ema_50': np.random.randn(200).cumsum() + 100  # Add EMA column
    }
    df = pd.DataFrame(data)
    
    print(f"Test data shape: {df.shape}")
    print(f"Required columns present: {all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ema_50'])}")
    
    # Capture warnings to check for SettingWithCopyWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            print("\n--- Test 1: Fix A - Attribute initialization order ---")
            print("Creating ContinuousTradingEnv...")
            
            # This should work without AttributeError (Fix A)
            env = ContinuousTradingEnv(
                df=df,
                initial_balance=10000,
                window_size=20,
                commission=0.001
            )
            
            print("✅ SUCCESS: Environment created without AttributeError")
            print(f"   - window_size initialized: {env.window_size}")
            print(f"   - initial_balance initialized: {env.initial_balance}")
            print(f"   - commission initialized: {env.commission}")
            print(f"   - Data processed successfully: {env.df.shape}")
            
            # Test basic functionality
            print("\n--- Test 2: Basic Environment Functionality ---")
            obs, info = env.reset()
            print(f"✅ Reset successful, observation shape: {obs.shape}")
            
            # Test a few steps
            for i in range(3):
                action = np.array([0.1])  # Buy action
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"   Step {i+1}: Reward = {reward:.6f}, Net worth = {info['net_worth']:.2f}")
            
            print("✅ Environment step function works correctly")
            
        except AttributeError as e:
            print(f"❌ FAILED: AttributeError still occurs: {e}")
            return False
        except Exception as e:
            print(f"❌ FAILED: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Check for SettingWithCopyWarning
    print("\n--- Test 3: Fix B - SettingWithCopyWarning Check ---")
    setting_with_copy_warnings = [warning for warning in w if issubclass(warning.category, pd.core.common.SettingWithCopyWarning)]
    
    if setting_with_copy_warnings:
        print("❌ FAILED: SettingWithCopyWarning still occurs:")
        for warning in setting_with_copy_warnings:
            print(f"   - {warning.message}")
        return False
    else:
        print("✅ SUCCESS: No SettingWithCopyWarning detected")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Fix A: Essential attributes initialized before _process_data call")
    print("✅ Fix B: .copy() prevents SettingWithCopyWarning")
    print("✅ Environment functions correctly without errors")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_fixes()
    if not success:
        exit(1)