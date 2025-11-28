import gymnasium as gym
import numpy as np
import pandas as pd
import wandb
from features import get_features

class TradingEnv(gym.Env):
    """
    Custom Gym environment for crypto trading with Volume Profile features.
    """
    def __init__(self, df, vp7_df, vp30_df, episode_length_days=30, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.vp7_df = vp7_df
        self.vp30_df = vp30_df
        self.episode_length = episode_length_days * 24  # hours
        self.initial_balance = initial_balance
        self.min_steps = 30 * 24  # to have VP data

        # Trading costs
        self.fees = 0.001  # 0.1% fee

        # Action space: position target from -1 (full short) to 1 (full long)
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        # Observation space: features vector
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(234,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.current_step = 500
        self.balance = self.initial_balance
        self.position = 0.0
        self.prev_position = 0.0
        self.current_price = self.df.iloc[self.current_step]['close']
        self.prev_price = self.current_price
        self.vp_poc_30d = self.vp30_df.iloc[self.current_step]['poc']
        return self._get_observation(), {}

    def step(self, action):
        prev_price = self.prev_price
        current_price = self.df.iloc[self.current_step]['close']
        self.current_price = current_price
        self.vp_poc_30d = self.vp30_df.iloc[self.current_step]['poc']
        old_portfolio = self.balance + self.prev_position * current_price

        # FIXED: Scale action to position (-1=full short, 0=flat, +1=full long)
        raw_action = action[0]  # SAC outputs array; take first element
        self.position = np.tanh(raw_action * 2.0) * 0.8  # Max 80% exposure for risk control
        self.position = np.clip(self.position, -0.99, 0.99)  # Avoid full 100% to prevent liquidation edge cases

        # Close old position + open new one (with fees)
        trade_cost = abs(self.position - self.prev_position) * current_price * 0.001  # 0.1% fee
        self.balance -= trade_cost
        self.prev_position = self.position

        # New portfolio value
        new_portfolio = self.balance + self.position * current_price
        portfolio_change = (new_portfolio - old_portfolio) / old_portfolio if old_portfolio != 0 else 0

        # === REAL PROFIT-SEEKING REWARD (the one that actually works) ===
        log_returns = np.log(current_price / prev_price)
        portfolio_return = self.position * log_returns   # PnL this step

        # 1. Raw PnL (positive = good)
        reward = portfolio_return * 25.0                 # scale up!

        # 2. Punish staying flat (this is the key!)
        reward -= abs(self.position) * 0.0001            # tiny holding cost
        reward -= (1.0 - abs(self.position)) * 0.002     # BIG penalty for being near zero!

        # 3. Bonus for being in high-volume-profile zones (your VP heatmaps)
        if self.current_price > self.vp_poc_30d:
            reward += 0.001 * abs(self.position)         # love the POC

        # 4. Only hard penalty if you actually blow up (optional)
        if self.balance < 500:  # near bankruptcy
            reward -= 20.0

        self.prev_price = current_price
        self.current_step += 1

        # === FINAL FIX: Gracefully stop when we run out of data ===
        if self.current_step >= len(self.df):
            # We have no more data → tell SB3 the episode is over (but do NOT reset anything)
            terminated = True      # ← This ends the episode cleanly
            truncated = False
            obs = self._get_observation()  # last valid observation
            print(f"\nReached end of data at step {self.current_step}. Ending episode cleanly.")
            return obs, reward, terminated, truncated, info

        # Normal case: continue training
        terminated = False
        truncated = False

        info = {
            "portfolio_value": self.balance + self.position * current_price,
            "position": self.position,
        }

        # Logging every 100 steps
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: Action={action[0]:.3f}, Position={self.position:.3f}, "
                  f"Reward={reward:.3f}, Portfolio={new_portfolio:.0f}")

        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_portfolio_value(self, price):
        return self.balance + self.position * price

    def _get_observation(self):
        t = self.df.index[self.current_step]
        obs = get_features(self.df, self.vp7_df, self.vp30_df, t)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros_like(obs)
        return obs

    def render(self, mode='human'):
        pass