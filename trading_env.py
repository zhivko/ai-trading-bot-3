import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Import your helper
from volume_profile import get_rolling_vp

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, lookback_window=30, vp_days=None):
        super(TradingEnv, self).__init__()

        # --- DATA PREP ---
        # Keep 'date' as a column so we can log it later
        self.raw_df = df.reset_index(drop=False) 
        self.df = self.raw_df.copy()
        
        # 1. Standard Features
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)
        self.df['volume_norm'] = (self.df['volume'] - self.df['volume'].mean()) / (self.df['volume'].std() + 1e-8)
        self.features = ['close_pct', 'volume_norm'] 
        
        # Fast access to standard features
        self.market_features = self.df[self.features].values.astype(np.float32)
        self.raw_prices = self.df['close'].values.astype(np.float32)
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # --- VOLUME PROFILE PRE-CALCULATION ---
        self.vp_days = vp_days if vp_days else [7, 30]
        self.vp_data = {}
        
        print(f"--- Initializing Environment (VP Days: {self.vp_days}) ---")
        
        for days in self.vp_days:
            vp_result = get_rolling_vp(self.raw_df, days, bin_percent=0.005)
            self.vp_data[days] = vp_result
            
        # Define Observation Space
        market_obs_size = self.lookback_window * len(self.features)
        account_obs_size = 2
        vp_obs_size = len(self.vp_days) * (3 + 100) # 3 scalars + 100 heatmap per VP
        total_obs_size = market_obs_size + account_obs_size + vp_obs_size
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )
        
        self.max_lookback = max([d * 24 for d in self.vp_days]) + self.lookback_window
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0
        
        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback

        return self._next_observation(), {}

    def _next_observation(self):
        # 1. Standard Window
        window_start = self.current_step - self.lookback_window
        std_features = self.market_features[window_start : self.current_step].flatten()
        
        # 2. Account Data
        current_price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * current_price) / self.initial_balance
        account_features = np.array([balance_norm, holdings_norm], dtype=np.float32)
        
        # 3. Volume Profile Data
        vp_features_list = []
        for days in self.vp_days:
            data = self.vp_data[days]
            
            poc = data['poc'][self.current_step]
            vah = data['vah'][self.current_step]
            val = data['val'][self.current_step]
            heatmap = data['heatmap'][self.current_step]
            
            if current_price > 0:
                dist_poc = (poc - current_price) / current_price
                dist_vah = (vah - current_price) / current_price
                dist_val = (val - current_price) / current_price
            else:
                dist_poc, dist_vah, dist_val = 0, 0, 0
                
            vp_features_list.extend([dist_poc, dist_vah, dist_val])
            vp_features_list.extend(heatmap)
            
        vp_features = np.array(vp_features_list, dtype=np.float32)
        
        # OPTIONAL: Add Noise here if you want to further reduce overfitting
        # noise = np.random.normal(0, 0.01, size=vp_features.shape)
        # vp_features = vp_features + (vp_features * noise)

        full_obs = np.concatenate((std_features, account_features, vp_features))
        return full_obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])
        
        # --- TRADE LOGIC (Higher Fee 0.1% to prevent overfitting) ---
        REALISTIC_FEE = 0.001
        trade_penalty = 0
        
        if action_val > 0.1: # Buy
            amount_to_invest = self.balance * action_val
            if amount_to_invest > 10: 
                shares_bought = amount_to_invest / current_price
                self.balance -= amount_to_invest
                self.shares_held += shares_bought
                trade_penalty = REALISTIC_FEE
            else:
                trade_penalty = 0.01 
        elif action_val < -0.1: # Sell
            shares_to_sell = self.shares_held * abs(action_val)
            if shares_to_sell * current_price > 10:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                trade_penalty = REALISTIC_FEE 
            else:
                trade_penalty = 0.01

        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # --- REWARD ---
        step_reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        reward = step_reward * 100 
        reward -= trade_penalty
        self.prev_net_worth = self.net_worth

        # --- TERMINATION ---
        terminated = False
        truncated = False
        if self.current_step >= len(self.df) - 1: truncated = True
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            reward = -10 

        obs = self._next_observation()
        
        # --- PREPARE LOGGING DATA ---
        primary_day = self.vp_days[0]
        heatmap = self.vp_data[primary_day]['heatmap'][self.current_step]
        
        # EXTRACT RAW VALUES FOR WANDB
        raw_poc = self.vp_data[primary_day]['poc'][self.current_step]
        raw_vah = self.vp_data[primary_day]['vah'][self.current_step]
        raw_val = self.vp_data[primary_day]['val'][self.current_step]
        
        # Calculate Real Price Bins for WandB
        window_size = primary_day * 24
        start_idx = max(0, self.current_step - window_size)
        window_prices = self.raw_prices[start_idx : self.current_step]
        
        if len(window_prices) > 0:
            min_p, max_p = np.min(window_prices), np.max(window_prices)
        else:
            min_p, max_p = current_price, current_price
            
        if min_p == max_p: price_bins = [min_p] * 100
        else: price_bins = np.linspace(min_p, max_p, 100).tolist()

        info = {
            "portfolio_value": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "reward": reward,
            "price": current_price,
            
            # Send raw values for charts
            "poc": raw_poc,
            "vah": raw_vah,
            "val": raw_val,
            
            "vp_heatmap": heatmap,
            "vp_bins": price_bins 
        }

        # Console Log with Date
        if self.current_step % 100 == 0:
            current_date = self.raw_df.iloc[self.current_step]['date']
            
            dist_pct = ((current_price - raw_poc) / raw_poc) * 100 if raw_poc != 0 else 0
            
            print(f"Step {self.current_step} [{current_date}]: P={current_price:.0f} | POC={raw_poc:.0f} ({dist_pct:+.2f}%) | Port={self.net_worth:.0f}")

        return obs, reward, terminated, truncated, info