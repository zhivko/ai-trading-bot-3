import pandas as pd
import pickle
from trading_env import TradingEnv
from volume_profile import get_rolling_vp

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('BTCUSDT_data.csv', index_col=0, parse_dates=True)

    # Load or compute VP
    vp7_file = 'BTCUSDT_vp7.pkl'
    vp30_file = 'BTCUSDT_vp30.pkl'

    try:
        with open(vp7_file, 'rb') as f:
            vp7_dict = pickle.load(f)
    except:
        vp7_dict = get_rolling_vp(df, 7)
        with open(vp7_file, 'wb') as f:
            pickle.dump(vp7_dict, f)

    try:
        with open(vp30_file, 'rb') as f:
            vp30_dict = pickle.load(f)
    except:
        vp30_dict = get_rolling_vp(df, 30)
        with open(vp30_file, 'wb') as f:
            pickle.dump(vp30_dict, f)

    # Create DataFrames from dicts
    vp7_df = pd.DataFrame({
        'poc': vp7_dict['poc'],
        'vah': vp7_dict['vah'],
        'val': vp7_dict['val'],
        'hvn': vp7_dict['hvn'],
        'lvn': vp7_dict['lvn'],
        'heatmap': [row for row in vp7_dict['heatmap']]
    }, index=df.index)

    vp30_df = pd.DataFrame({
        'poc': vp30_dict['poc'],
        'vah': vp30_dict['vah'],
        'val': vp30_dict['val'],
        'hvn': vp30_dict['hvn'],
        'lvn': vp30_dict['lvn'],
        'heatmap': [row for row in vp30_dict['heatmap']]
    }, index=df.index)

    # Create env
    env = TradingEnv(df, vp7_df, vp30_df)

    # Reset
    obs = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated and step_count < 10:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        print(f"Step {step_count}: portfolio={info['portfolio_value']:.2f}, reward={reward:.4f}, phase={info['current_phase']}, terminated={terminated}")

    print(f"Final: step {step_count}, portfolio={info['portfolio_value']:.2f}, terminated={terminated}, truncated={truncated}")