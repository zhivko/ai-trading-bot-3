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
        self.current_step = 500  # Start at 500 for VP data availability
        self.train_step = 0  # Track actual training steps (starts from 0)
        self.balance = self.initial_balance
        self.holdings = 0.0  # Actual BTC units held (positive=long, negative=short)
        self.target_position = 0.0  # Target position fraction (-1 to +1)
        self.current_price = self.df.iloc[self.current_step]['close']
        self.prev_price = self.current_price
        self.vp_poc_30d = self.vp30_df.iloc[self.current_step]['poc']
        return self._get_observation(), {}

    def step(self, action):
        prev_price = self.prev_price
        current_price = self.df.iloc[self.current_step]['close']
        self.current_price = current_price
        self.vp_poc_30d = self.vp30_df.iloc[self.current_step]['poc']
        
        # Calculate current portfolio value BEFORE any trades
        old_portfolio = self.balance + self.holdings * current_price

        # Scale action to target position fraction (-1=full short, 0=flat, +1=full long)
        raw_action = action[0]  # SAC outputs array; take first element
        new_target = np.tanh(raw_action * 2.0) * 0.8  # Max 80% exposure for risk control
        new_target = np.clip(new_target, -0.99, 0.99)
        
        # Calculate target holdings in BTC units based on portfolio value
        # target_position is fraction of portfolio to hold in BTC
        target_value = new_target * old_portfolio  # Dollar value to hold in BTC
        target_holdings = target_value / current_price  # BTC units to hold
        
        # Calculate trade size and apply fees
        trade_size = abs(target_holdings - self.holdings)  # BTC units traded
        trade_value = trade_size * current_price  # Dollar value traded
        trade_cost = trade_value * self.fees  # 0.1% fee
        
        # Execute trade: adjust balance for buying/selling BTC
        btc_change = target_holdings - self.holdings
        self.balance -= btc_change * current_price  # Pay for BTC (or receive cash if selling)
        self.balance -= trade_cost  # Pay trading fees
        self.holdings = target_holdings
        self.target_position = new_target

        # Calculate new portfolio value AFTER trades
        new_portfolio = self.balance + self.holdings * current_price

        # Debug logging for low portfolio values
        if new_portfolio <= 10:
            print(f"Portfolio <=10: {new_portfolio:.6f}, balance={self.balance:.6f}, holdings={self.holdings:.6f}, price={current_price:.2f}")

        # === REWARD CALCULATION ===
        log_returns = np.log(current_price / prev_price) if prev_price > 0 else 0
        portfolio_return = self.target_position * log_returns  # PnL this step

        # 1. Raw PnL (positive = good)
        reward = portfolio_return * 25.0

        # 2. Punish staying flat
        reward -= abs(self.target_position) * 0.0001  # tiny holding cost
        reward -= (1.0 - abs(self.target_position)) * 0.002  # penalty for being near zero

        # 3. Bonus for being in high-volume-profile zones
        if self.current_price > self.vp_poc_30d:
            reward += 0.001 * abs(self.target_position)

        # 4. Check for bankruptcy
        terminated = False
        if new_portfolio <= 0.01:
            reward -= 50.0  # Large penalty for going bankrupt
            terminated = True
            print(f"\n[BANKRUPT] Step {self.train_step}: Portfolio went to {new_portfolio:.2f}")
        elif new_portfolio < self.initial_balance * 0.1:  # Below 10% of initial
            reward -= 10.0  # Penalty for near-bankruptcy

        self.prev_price = current_price
        self.current_step += 1
        self.train_step += 1

        # === WRAP AROUND DATA ===
        truncated = False
        if self.current_step >= len(self.df):
            self.current_step = 500  # Reset to min_steps for VP data
            self.prev_price = self.df.iloc[self.current_step]['close']
            print(f"\n[Train step {self.train_step}] Data wrapped around. Portfolio: {new_portfolio:.0f}")

        info = {
            "portfolio_value": new_portfolio,
            "position": self.target_position,
            "holdings": self.holdings,
            "balance": self.balance,
        }

        # Logging
        if self.train_step % 100 == 0:
            print(f"Step {self.train_step}: Action={action[0]:.3f}, Position={self.target_position:.3f}, Holdings={self.holdings:.6f}, Reward={reward:.3f}, Portfolio={new_portfolio:.0f}")

        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_portfolio_value(self, price):
        return self.balance + self.holdings * price

    def _get_observation(self):
        t = self.df.index[self.current_step]
        obs = get_features(self.df, self.vp7_df, self.vp30_df, t)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros_like(obs)
        return obs

    def render(self, mode='human'):
        pass