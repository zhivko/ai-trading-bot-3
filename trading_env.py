import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, lookback_window=30):
        super(TradingEnv, self).__init__()

        # 1. Pre-process Data: Normalize
        self.raw_df = df.reset_index(drop=True)
        self.df = self.raw_df.copy()
        
        # Calculate features (Normalized)
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)
        # Normalize volume
        self.df['volume_norm'] = (self.df['volume'] - self.df['volume'].mean()) / (self.df['volume'].std() + 1e-8)
        
        self.features = ['close_pct', 'volume_norm'] 
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Action: -1 (Sell All) to +1 (Buy All)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation: Market Features + Account State
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, len(self.features) + 2), 
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = self.lookback_window
        
        # Random start
        if len(self.df) > 2000:
            self.current_step = np.random.randint(self.lookback_window, len(self.df) - 1000)

        return self._next_observation(), {}

    def _next_observation(self):
        # 1. Market Data
        frame = self.df.iloc[self.current_step - self.lookback_window : self.current_step][self.features]
        obs = frame.values
        
        # 2. Account Data (Normalized)
        balance_norm = self.balance / self.initial_balance
        # Calculate current value of holdings
        current_holdings_value = self.shares_held * self.raw_df.iloc[self.current_step]['close']
        holdings_norm = current_holdings_value / self.initial_balance
        
        # Stack to create (Window, Features+2)
        account_info = np.array([[balance_norm, holdings_norm]] * self.lookback_window)
        full_obs = np.hstack((obs, account_info))
        
        return full_obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Get raw price for calculations
        current_price = self.raw_df.iloc[self.current_step]['close']
        action_val = float(action[0])
        
        # --- TRADE EXECUTION ---
        trade_penalty = 0
        
        if action_val > 0.1: # BUY
            # Amount to invest: % of available cash
            amount_to_invest = self.balance * action_val
            if amount_to_invest > 10: 
                shares_bought = amount_to_invest / current_price
                self.balance -= amount_to_invest
                self.shares_held += shares_bought
                trade_penalty = 0.0005 # Fee
            else:
                trade_penalty = 0.01 # Penalty for invalid buy (no cash)
                
        elif action_val < -0.1: # SELL
            # Shares to sell: % of held shares
            shares_to_sell = self.shares_held * abs(action_val)
            if shares_to_sell * current_price > 10:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                trade_penalty = 0.0005 # Fee
            else:
                trade_penalty = 0.01 # Penalty for invalid sell (no shares)

        # --- UPDATE STATE ---
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # --- REWARD (Incremental) ---
        # Reward is % change in portfolio value
        step_reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        reward = step_reward * 100  # Scale up for stability
        reward -= trade_penalty
        
        self.prev_net_worth = self.net_worth

        # --- TERMINATION ---
        terminated = False
        truncated = False
        
        if self.current_step >= len(self.df) - 1:
            truncated = True
            
        # Hard Bankruptcy Stop (50% loss)
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            reward = -10 # Final penalty

        obs = self._next_observation()
        
        # --- LOGGING FOR WANDB ---
        # Everything here will show up in WandB
        info = {
            "portfolio_value": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "reward": reward,
            "price": current_price
        }

        # Console logging for sanity check
        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step}: P={current_price:.2f}, Act={action_val:.2f}, Port={self.net_worth:.2f}")

        return obs, reward, terminated, truncated, info