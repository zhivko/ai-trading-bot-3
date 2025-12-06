import pandas as pd
import numpy as np
from enhanced_trading_env import EnhancedTradingEnv
from volume_profile import get_rolling_vp

def main():
    # Load data (assuming BTCUSDT_data.csv exists)
    df = pd.read_csv('BTCUSDT_data.csv')
    df.columns = df.columns.str.lower()
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Create env (skip vp for simplicity)
    env = EnhancedTradingEnv(
        df=df,
        initial_balance=10000,
        lookback_window=50,
        precalculated_vp={},  # Skip vp calculation
        vp_days=[3,7],  # But since precalculated is empty, it won't calculate
        vp_bins=40
    )

    # Reset
    obs, info = env.reset()
    print(f"Reset obs shape: {obs.shape}")
    print(f"Expected shape: {env.observation_space.shape}")

    # Step with large action to test clipping
    action = np.array([3.5])  # Test clipping
    obs_next, reward, terminated, truncated, info = env.step(action)
    print(f"Step obs shape: {obs_next.shape}")
    print(f"Info keys: {list(info.keys())}")
    print(f"Action taken: {info.get('action', 0)}")  # Should be clipped to 1.0
    print(f"Trade executed: {info.get('trade', False)}")
    print(f"Net worth: {info.get('portfolio_value', 0)}")
    print(f"Shares held: {info.get('shares_held', 0)}")
    print(f"VP heatmap shape: {info.get('vp_heatmap', np.array([])).shape if info.get('vp_heatmap') is not None else 'None'}")

    # Check consistency
    assert obs.shape == env.observation_space.shape, f"Shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    assert obs_next.shape == env.observation_space.shape, f"Shape mismatch: {obs_next.shape} vs {env.observation_space.shape}"
    assert len(info) > 0, "Info dict empty"
    assert 'shares_held' in info, "Missing shares_held in info"
    assert 'vp_heatmap' in info, "Missing vp_heatmap in info"

    print("Verification successful: Shapes consistent, no errors.")

if __name__ == '__main__':
    main()