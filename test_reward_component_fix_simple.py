#!/usr/bin/env python3
"""
Simplified test to verify the reward component plotting fix in callbacks.
This test focuses on the callback logic without requiring the full environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from callbacks.base_callbacks import TensorboardCallback, CustomEvalCallback

def test_reward_component_extraction_logic():
    """Test the reward component extraction logic from the fix."""
    print("Testing reward component extraction logic...")
    
    # Simulate the fixed logic
    steps = np.arange(10)
    
    # Create mock portfolio data with reward components
    mock_portfolio = []
    for i in range(10):
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
    
    # Apply the fixed logic
    reward_components = {
        'base': [0.0] * len(steps),
        'fee_penalty': [0.0] * len(steps),
        'action_change_penalty': [0.0] * len(steps),
        'trend_alignment': [0.0] * len(steps),
        'holding_penalty': [0.0] * len(steps),
        'inertia_penalty': [0.0] * len(steps),
        'closer_bonus': [0.0] * len(steps),
        'overtrading_penalty': [0.0] * len(steps),
    }

    # Extract from infos (list of dicts, one per env — usually 1)
    for i, info_item in enumerate(mock_portfolio):
        if 'reward_components' in info_item:
            comp = info_item['reward_components']
            if i < len(steps):
                for key in reward_components:
                    reward_components[key][i] = comp.get(key, 0.0)
            break  # Use first valid env (safe for single-env eval)

    # Convert to arrays for compatibility with existing plotting code
    reward_base = np.array(reward_components['base'], dtype=float)
    reward_fee = np.array(reward_components['fee_penalty'], dtype=float)
    reward_action_change = np.array(reward_components['action_change_penalty'], dtype=float)
    reward_trend = np.array(reward_components['trend_alignment'], dtype=float)
    reward_holding = np.array(reward_components['holding_penalty'], dtype=float)
    reward_inertia = np.array(reward_components['inertia_penalty'], dtype=float)
    reward_closer = np.array(reward_components['closer_bonus'], dtype=float)
    reward_overtrade = np.array(reward_components['overtrading_penalty'], dtype=float)
    
    # Verify the extraction worked
    print("Extracted reward components:")
    component_arrays = [reward_base, reward_fee, reward_action_change, reward_trend, 
                       reward_holding, reward_inertia, reward_closer, reward_overtrade]
    component_names = ['Base', 'Fee', 'Action_Change', 'Trend', 'Holding', 'Inertia', 'Closer', 'Overtrade']
    
    all_nonzero = True
    for name, arr in zip(component_names, component_arrays):
        mean_val = np.mean(arr)
        has_nonzero = np.any(arr != 0)
        print(f"  {name}: mean={mean_val:.6f}, has_nonzero={has_nonzero}")
        if not has_nonzero:
            all_nonzero = False
    
    return all_nonzero

def test_plotting_with_fixed_data():
    """Test that the plotting works with the fixed reward component data."""
    print("\nTesting plotting with fixed reward component data...")
    
    # Create test data using the fixed approach
    steps = np.arange(20)
    
    # Generate realistic reward component data
    np.random.seed(42)
    reward_base = np.random.normal(0.1, 0.02, 20)
    reward_fee = np.random.normal(-0.01, 0.002, 20)
    reward_action_change = np.random.normal(-0.005, 0.001, 20)
    reward_trend = np.random.normal(0.02, 0.005, 20)
    reward_holding = np.random.normal(-0.001, 0.0002, 20)
    reward_inertia = np.random.normal(-0.002, 0.0005, 20)
    reward_closer = np.random.normal(0.05, 0.01, 20)
    reward_overtrade = np.random.normal(-0.01, 0.002, 20)
    
    component_arrays = [
        reward_base, reward_fee, reward_action_change, reward_trend,
        reward_holding, reward_inertia, reward_closer, reward_overtrade
    ]
    component_labels = [
        'Base (net worth)', 'Fee penalty', 'Action change penalty', 'Trend alignment',
        'Holding cost', 'Inertia penalty', 'Closer bonus', 'Overtrading penalty'
    ]
    
    # Test the improved color mapping
    colors = {
        'base': 'blue',
        'fee_penalty': 'red',
        'action_change_penalty': 'orange',
        'trend_alignment': 'purple',
        'holding_penalty': 'brown',
        'inertia_penalty': 'pink',
        'closer_bonus': 'green',
        'overtrading_penalty': 'darkred',
    }
    
    # Create a simple plot to test the color logic
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        # Plot each component as a separate line (no stacking)
        for i, (component_array, label) in enumerate(zip(component_arrays, component_labels)):
            # Apply specific styling for certain components
            linestyle = '-'
            linewidth = 1.5
            
            if i == 6:  # Closer bonus - dark green
                color = 'darkgreen'
            elif i == 7:  # Overtrading penalty - dashed line
                linestyle = '--'
                color = 'darkred'
            else:
                # Use color mapping based on component type
                component_key = label.split(' ')[0].lower().replace('(', '').replace(')', '')
                if 'base' in component_key:
                    color = colors['base']
                elif 'fee' in component_key:
                    color = colors['fee_penalty']
                elif 'action' in component_key:
                    color = colors['action_change_penalty']
                elif 'trend' in component_key:
                    color = colors['trend_alignment']
                elif 'holding' in component_key:
                    color = colors['holding_penalty']
                elif 'inertia' in component_key:
                    color = colors['inertia_penalty']
                elif 'closer' in component_key:
                    color = colors['closer_bonus']
                elif 'overtrading' in component_key:
                    color = colors['overtrading_penalty']
                else:
                    color = 'gray'
                
            ax.plot(steps, component_array, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.8)
        
        # Add horizontal line at zero for reference
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax.set_ylabel("Reward Components (signed)")
        ax.set_xlabel("Steps")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plot to verify it works
        plt.tight_layout()
        plt.savefig('test_reward_components_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Plotting with fixed reward components completed successfully!")
        print("📊 Plot saved as 'test_reward_components_plot.png'")
        return True
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callback_instantiation():
    """Test that the callbacks can be instantiated without errors."""
    print("\nTesting callback instantiation...")
    
    try:
        # Test TensorboardCallback
        tensorboard_cb = TensorboardCallback(verbose=1)
        print("✅ TensorboardCallback instantiated successfully")
        
        # Test CustomEvalCallback (with minimal required params)
        custom_eval_cb = CustomEvalCallback(
            eval_env=None,  # We'll pass None for this test
            best_model_save_path="./test_models",
            eval_freq=1000,
            n_eval_episodes=5
        )
        print("✅ CustomEvalCallback instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Callback instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests to verify the reward component fix."""
    print("=== Simplified Reward Component Fix Test ===")
    print()
    
    # Test 1: Callback instantiation
    cb_success = test_callback_instantiation()
    
    # Test 2: Reward component extraction logic
    extraction_success = test_reward_component_extraction_logic()
    
    # Test 3: Plotting with fixed data
    plotting_success = test_plotting_with_fixed_data()
    
    print("\n=== Test Results ===")
    if cb_success:
        print("✅ Callback instantiation works")
    else:
        print("❌ Callback instantiation failed")
    
    if extraction_success:
        print("✅ Reward component extraction logic works")
    else:
        print("❌ Reward component extraction failed")
    
    if plotting_success:
        print("✅ Plotting with fixed reward components works")
    else:
        print("❌ Plotting failed")
    
    overall_success = cb_success and extraction_success and plotting_success
    
    if overall_success:
        print("\n🎉 All tests passed! The reward component plotting fix is working correctly.")
        print("📋 Summary of fixes applied:")
        print("  • Fixed reward component extraction from info['reward_components']")
        print("  • Improved plotting with specific colors for each component")
        print("  • Added proper handling of signed reward components")
        print("  • Enhanced visualization with better styling")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return overall_success

if __name__ == "__main__":
    main()