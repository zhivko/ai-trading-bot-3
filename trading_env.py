import gymnasium as gym
import numpy as np
import pandas as pd
import wandb
from features import get_features

class TradingEnv(gym.Env):
    """
    Custom Gym environment for crypto trading with Volume Profile features.
    """
    def __init__(self, df, vp7_df, vp30_df, episode_length_days=30, initial_cash=10000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.vp7_df = vp7_df
        self.vp30_df = vp30_df
        self.episode_length = episode_length_days * 24  # hours
        self.initial_cash = initial_cash
        self.min_steps = 30 * 24  # to have VP data

        # Trading costs
        self.fees = 0.0010
        self.slippage = 0.0005

        # Action space: position target from -1 (full short) to 1 (full long)
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        # Observation space: features vector
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(234,), dtype=np.float32)

        self.reset()

    def reset(self, **kwargs):
        # Start from the earliest point with enough history for full backtest
        self.start_step = self.min_steps
        self.current_step = self.start_step
        self.position = 0.0  # -1 to 1
        self.cash = self.initial_cash
        self.entry_price = 0.0
        self.done = False

        # Tracking variables
        self.previous_action = 0.0
        self.holding_bars = 0
        self.position_age = 0
        self.returns = []

        obs = self._get_obs()
        return obs, {}  # Return observation and info dict as per gymnasium interface

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")
        action = np.clip(action[0], -1.0, 1.0)
        action = np.sign(action) * max(abs(action), 0.3)  # min 30% size

        # Get current price
        close = self.df.iloc[self.current_step]['close']

        # Calculate previous portfolio value
        prev_value = self._calculate_portfolio_value(close)

        # Old position
        old_position = self.position

        # Update position and entry price
        position_changed = abs(action - self.position) > 1e-6
        if position_changed:
            self.position = action
            self.entry_price = close
            self.holding_bars = 0
        else:
            self.holding_bars += 1

        # Move to next step
        self.current_step += 1
        next_close = self.df.iloc[self.current_step]['close'] if self.current_step < len(self.df) else close

        # Calculate new portfolio value
        current_value = self._calculate_portfolio_value(next_close)

        # Reward: change in portfolio value
        reward = current_value - prev_value

        # Subtract trading costs
        if position_changed:
            traded_volume = abs(action - old_position) * self.cash
            cost = traded_volume * (self.fees + self.slippage)
            reward -= cost

        # New fee for action change
        if action != self.previous_action:
            reward -= abs(action - self.previous_action) * self.fees * abs(old_position)

        # Direction change penalty
        if np.sign(action) != np.sign(old_position) and old_position != 0 and action != 0:
            reward -= 15.0

        # Minimum holding penalty
        if action == 0 and old_position != 0 and self.holding_bars < 6:
            reward -= 20.0

        # Volume profile bonuses/penalties
        t = self.df.index[self.current_step - 1]
        vp7 = self.vp7_df.loc[t]
        poc = vp7['poc'] if not pd.isna(vp7['poc']) else close
        vah = vp7['vah'] if not pd.isna(vp7['vah']) else close
        val = vp7['val'] if not pd.isna(vp7['val']) else close

        dist_to_poc = abs(close - poc) / close if close != 0 else 0

        # VP bonuses
        if self.position > 0 and close <= val * 1.008:
            reward += 12.0
        if self.position < 0 and close >= vah * 0.992:
            reward += 12.0

        # Penalty for fighting POC with big size
        if abs(self.position) > 0.5 and dist_to_poc > 0.02:
            reward -= 8.0

        # Update position_age
        if abs(self.position) > 0:
            self.position_age += 1

        # Update previous_action
        self.previous_action = action

        # Check if episode done
        self.done = (self.current_step >= len(self.df) - 1) or ((self.current_step - self.start_step) >= self.episode_length)

        obs = self._get_obs()
        # Return gymnasium format: (obs, reward, terminated, truncated, info)
        return obs, reward, self.done, False, {}

    def _calculate_portfolio_value(self, price):
        if self.position == 0:
            return self.cash
        elif self.position > 0:  # Long
            return self.cash + self.position * self.cash * (price / self.entry_price - 1)
        else:  # Short
            return self.cash + abs(self.position) * self.cash * (self.entry_price / price - 1)

    def _get_obs(self):
        t = self.df.index[self.current_step]
        return get_features(self.df, self.vp7_df, self.vp30_df, t)

    def render(self, mode='human'):
        pass