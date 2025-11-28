import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, lookback_window=30):
        super(TradingEnv, self).__init__()

        # 1. Pre-process Data: Normalize immediately
        # We don't want raw prices (e.g. 50000). We want % change (e.g. 0.01)
        self.raw_df = df.reset_index(drop=True)
        self.df = self.raw_df.copy()
        
        # Calculate log returns or pct_change for normalization
        # This is crucial for the Neural Network to learn patterns
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)
        self.df['volume_norm'] = (self.df['volume'] - self.df['volume'].mean()) / (self.df['volume'].std() + 1e-8)
        
        # Define the features we will give the AI
        self.features = ['close_pct', 'volume_norm'] 
        # Add other technical indicators here if you have them (RSI, MACD, etc) normalized!

        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space shape depends on number of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, len(self.features) + 2), # +2 for balance/position info
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance # Track previous step for reward
        self.shares_held = 0
        self.current_step = self.lookback_window
        
        # Random start to prevent overfitting
        if len(self.df) > 2000:
            self.current_step = np.random.randint(self.lookback_window, len(self.df) - 1000)

        return self._next_observation(), {}

    def _next_observation(self):
        # 1. Get Market Data Window
        frame = self.df.iloc[self.current_step - self.lookback_window : self.current_step][self.features]
        obs = frame.values
        
        # 2. Append Account Info (Normalized) to EACH timestamp in the window
        # The bot needs to know: "Do I have money?" and "Do I have shares?"
        # We normalize balance by dividing by initial_balance (approx)
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * self.raw_df.iloc[self.current_step]['close']) / self.initial_balance
        
        # Create a matching shape to stack
        account_info = np.array([[balance_norm, holdings_norm]] * self.lookback_window)
        
        # Combine Market Data + Account Data
        # Final shape: (30, features + 2)
        full_obs = np.hstack((obs, account_info))
        
        return full_obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Get Current Price from RAW data (for calculation)
        current_price = self.raw_df.iloc[self.current_step]['close']
        
        action_val = float(action[0])
        
        # --- EXECUTE TRADE ---
        # Logic: 
        # Action > 0: Buy with X% of CASH
        # Action < 0: Sell X% of SHARES
        
        trade_penalty = 0
        
        if action_val > 0.1: # Buy threshold
            # Example: Action 0.5 = Buy with 50% of available cash
            amount_to_invest = self.balance * action_val
            if amount_to_invest > 10: # Minimum trade size
                shares_bought = amount_to_invest / current_price
                self.balance -= amount_to_invest
                self.shares_held += shares_bought
                trade_penalty = 0.0005 # Small fee simulation (0.05%)
            else:
                # Penalty for trying to buy with no money (Teaches it to hold or sell)
                trade_penalty = 0.01 
                
        elif action_val < -0.1: # Sell threshold
            # Example: Action -0.5 = Sell 50% of held shares
            shares_to_sell = self.shares_held * abs(action_val)
            if shares_to_sell * current_price > 10:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                trade_penalty = 0.0005
            else:
                 # Penalty for trying to sell nothing
                trade_penalty = 0.01

        # Calculate Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # --- REWARD CALCULATION (CRITICAL CHANGE) ---
        
        # 1. Step Reward: Pure percentage change from LAST STEP
        # This rewards "Making Money Now", not "Recovering from past losses"
        step_reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        
        # 2. Scale it up: RL works better with numbers like 1.0 or -1.0, not 0.0001
        reward = step_reward * 100 
        
        # 3. Apply Penalties
        reward -= trade_penalty

        # 4. Update previous net worth
        self.prev_net_worth = self.net_worth

        # --- TERMINATION LOGIC ---
        terminated = False
        
        # Stop if we ran out of data
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
        # Stop if bankrupt (lose 50% of money) - Tighter leash
        if self.net_worth < (self.initial_balance * 0.5):
            reward = -10 # Big penalty
            terminated = True
            print(f"BANKRUPT at step {self.current_step}. Net Worth: {self.net_worth}")

        obs = self._next_observation()
        info = {'net_worth': self.net_worth}
        
        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step}: Act={action_val:.2f}, Bal={self.balance:.0f}, Held={self.shares_held:.4f}, Net={self.net_worth:.0f}, Rwrd={reward:.4f}")

        return obs, reward, terminated, False, info