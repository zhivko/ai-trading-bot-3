import pandas as pd
import pickle
from trading_env import TradingEnv
from volume_profile import get_rolling_vp

# Load data
df = pd.read_csv('BTCUSDT_data.csv', index_col=0, parse_dates=True)

# Load or compute VP
vp7_file = 'BTC/USDT_vp7.pkl'
vp30_file = 'BTC/USDT_vp30.pkl'

try:
    with open(vp7_file, 'rb') as f:
        vp7_df = pickle.load(f)
except:
    vp7_df = get_rolling_vp(df, 7)
    with open(vp7_file, 'wb') as f:
        pickle.dump(vp7_df, f)

try:
    with open(vp30_file, 'rb') as f:
        vp30_df = pickle.load(f)
except:
    vp30_df = get_rolling_vp(df, 30)
    with open(vp30_file, 'wb') as f:
        pickle.dump(vp30_df, f)

# Create env
env = TradingEnv(df, vp7_df, vp30_df)

# Reset
obs = env.reset()
terminated = False
truncated = False
step_count = 0

while not terminated and not truncated and step_count < 1000:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    if step_count % 100 == 0:
        print(f"Step {step_count}: portfolio={info['portfolio_value']:.2f}, terminated={terminated}")

print(f"Final: step {step_count}, portfolio={info['portfolio_value']:.2f}, terminated={terminated}, truncated={truncated}")