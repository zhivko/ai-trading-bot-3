import sys
sys.path.append('.')
from callbacks.base_callbacks import TensorboardCallback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

# Create callback instance
cb = TensorboardCallback()
# Populate dummy data
n = 50
cb.ep_prices = np.random.randn(n).cumsum() + 100
cb.ep_emas = cb.ep_prices - 5 + np.random.randn(n) * 2
cb.ep_actions = np.random.randn(n) * 0.5
cb.ep_rewards = np.random.randn(n) * 0.1
cb.ep_portfolio = [{'net_worth': 1000 + i*10, 'trade_executed': False, 'panic_close': False, 'action': a} for i, a in enumerate(cb.ep_actions)]
cb.ep_dates = list(range(n))
# Reward components (must sum to total reward approximately)
cb.ep_reward_base = np.random.randn(n) * 0.05
cb.ep_reward_fee = np.full(n, -0.001)
cb.ep_reward_action_change = np.random.randn(n) * 0.01
cb.ep_reward_trend = np.random.randn(n) * 0.02
cb.ep_reward_holding = np.full(n, -0.0005)
cb.ep_reward_inertia = np.random.randn(n) * 0.005
cb.ep_reward_closer = np.random.randn(n) * 0.03
cb.ep_reward_overtrade = np.full(n, -0.002)
cb.ep_reward_episode = np.zeros(n)

# Call the plotting method (should not raise exceptions)
try:
    cb._plot_regime_chart()
    print("Plotting succeeded (no exception)")
except Exception as e:
    print(f"Plotting failed with error: {e}")
    import traceback
    traceback.print_exc()