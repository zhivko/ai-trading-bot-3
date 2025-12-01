import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import hashlib
import os
import pickle

# Import the existing calculation logic from your repo
from volume_profile import get_rolling_vp

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=30, vp_days=None):
        super(TradingEnv, self).__init__()
        
        # --- DATA PREP ---
        self.raw_df = df.reset_index(drop=False) 
        self.df = self.raw_df.copy()
        
        # 1. Standard Features (Matches your repo logic)
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)
        
        vol_std = self.df['volume'].std()
        self.df['volume_norm'] = (self.df['volume'] - self.df['volume'].mean()) / (vol_std if vol_std != 0 else 1)

        # 2. INDICATORS
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi_norm'] = self.df['rsi'] / 100.0

        # Stoch RSI
        min_rsi = self.df['rsi'].rolling(window=14).min()
        max_rsi = self.df['rsi'].rolling(window=14).max()
        div = max_rsi - min_rsi
        div[div == 0] = 1e-8
        self.df['stoch_rsi'] = (self.df['rsi'] - min_rsi) / div
        self.df['stoch_rsi_norm'] = self.df['stoch_rsi'].fillna(0.5) 

        # MACD
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_norm'] = self.df['macd'] / self.df['close']
        self.df['macd_sig_norm'] = self.df['macd_signal'] / self.df['close']

        self.features = ['close_pct', 'volume_norm', 'rsi_norm', 'stoch_rsi_norm', 'macd_norm', 'macd_sig_norm']
        self.df.fillna(0, inplace=True)
        
        self.market_features = self.df[self.features].values.astype(np.float32)
        self.raw_prices = self.df['close'].values.astype(np.float32)
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # --- VOLUME PROFILE (Cached Wrapper) ---
        self.vp_days = vp_days if vp_days else [7, 30]
        self.vp_data = {}
        
        # Create Hash of data to ensure cache is valid for this specific CSV
        # We hash the close and volume columns
        data_to_hash = self.df[['close', 'volume']].values.tobytes()
        self.data_hash = hashlib.md5(data_to_hash).hexdigest()
        
        self.cache_dir = "vp_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        print(f"--- Initializing Environment (VP Days: {self.vp_days}) ---")
        
        for days in self.vp_days:
            # Matches your file pattern: vp_win30_f389...pkl
            filename = f"vp_win{days}_{self.data_hash}.pkl"
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(filepath):
                print(f"⚡ Loading cached VP for {days} days: {filename}")
                try:
                    with open(filepath, 'rb') as f:
                        self.vp_data[days] = pickle.load(f)
                except Exception as e:
                    print(f"⚠️ Error loading cache {filename}: {e}. Recalculating.")
                    # Fallback to calculation
                    self.vp_data[days] = get_rolling_vp(self.raw_df, days, bin_percent=0.005)
            else:
                print(f"⚙️ Calculating VP for {days} days (Cache miss)...")
                # CALL THE FUNCTION FROM YOUR REPO
                vp_result = get_rolling_vp(self.raw_df, days, bin_percent=0.005)
                self.vp_data[days] = vp_result
                
                # Save to cache
                try:
                    with open(filepath, 'wb') as f:
                        pickle.dump(vp_result, f)
                    print(f"💾 Saved cache: {filename}")
                except Exception as e:
                    print(f"⚠️ Failed to save cache: {e}")
        
        # Define Observation Space
        # Calculate size: Market Features + Account(2) + VP((3+Bins)*NumTimeframes)
        # Assuming get_rolling_vp returns heatmap of size 100 (check your volume_profile.py if different)
        sample_heatmap_len = len(self.vp_data[self.vp_days[0]]['heatmap'][0])
        
        market_obs_size = self.lookback_window * len(self.features)
        account_obs_size = 2
        vp_obs_size = len(self.vp_days) * (3 + sample_heatmap_len) 
        total_obs_size = market_obs_size + account_obs_size + vp_obs_size
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32)
        
        self.max_lookback = max(max([d * 24 for d in self.vp_days]), 30) + self.lookback_window
        
        # State variables
        self.phase = 1
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.prev_action = 0
        self.trade_count = 0

    def set_phase(self, new_phase):
        self.phase = new_phase

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.prev_action = 0
        self.history_net_worth = [self.initial_balance]
        
        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback
            
        return self._next_observation(), {}

    def _next_observation(self):
        # 1. Market Features (Flattened Window)
        window_start = self.current_step - self.lookback_window
        std_features = self.market_features[window_start : self.current_step].flatten()
        
        # 2. Account Features
        current_price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * current_price) / self.initial_balance
        account_features = np.array([balance_norm, holdings_norm], dtype=np.float32)
        
        # 3. VP Features
        vp_features_list = []
        for days in self.vp_days:
            data = self.vp_data[days]
            poc = data['poc'][self.current_step]
            vah = data['vah'][self.current_step]
            val = data['val'][self.current_step]
            heatmap = data['heatmap'][self.current_step]
            
            if current_price > 0 and poc > 0:
                dist_poc = (poc - current_price) / current_price
                dist_vah = (vah - current_price) / current_price
                dist_val = (val - current_price) / current_price
            else:
                dist_poc, dist_vah, dist_val = 0, 0, 0
                
            vp_features_list.extend([dist_poc, dist_vah, dist_val])
            vp_features_list.extend(heatmap)
            
        vp_features = np.array(vp_features_list, dtype=np.float32)
        
        return np.concatenate((std_features, account_features, vp_features)).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])
        
        # --- TRADE LOGIC ---
        # 1. ENCOURAGE LONGER TRADES: Higher Fee (0.25%)
        REALISTIC_FEE = 0.0025 
        trade_penalty = 0
        
        if abs(action_val - self.prev_action) > 0.1:
            self.trade_count += 1
            self.prev_action = action_val

        # Execute Trade
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
        self.history_net_worth.append(self.net_worth)
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        step_reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        reward = step_reward * 100 
        reward -= trade_penalty

        # --- 2. ENCOURAGE RSI & STOCH RSI (Reward Shaping) ---
        cur_rsi = self.df.iloc[self.current_step]['rsi']
        cur_stoch = self.df.iloc[self.current_step]['stoch_rsi']
        indicator_bonus = 0.0

        # Reward BUYING when Oversold
        if action_val > 0.2:
            if cur_rsi < 30: indicator_bonus += 0.5 
            if cur_stoch < 0.2: indicator_bonus += 0.3
        # Reward SELLING when Overbought
        elif action_val < -0.2:
            if cur_rsi > 70: indicator_bonus += 0.5 
            if cur_stoch > 0.8: indicator_bonus += 0.3
        
        reward += indicator_bonus
        # --- END REWARD SHAPING ---

        if self.phase >= 2 and step_reward < 0:
            reward -= abs(step_reward) * 5.0
        if self.phase >= 3:
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            reward -= (drawdown * 0.1)

        self.prev_net_worth = self.net_worth

        terminated = False
        truncated = False
        if self.current_step >= len(self.df) - 1: truncated = True
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            reward = -10 

        obs = self._next_observation()
        
        # Info for Backtesting / Visualization
        primary_day = self.vp_days[0]
        # Safety for index
        idx_safe = min(self.current_step, len(self.vp_data[primary_day]['heatmap']) - 1)
        heatmap = self.vp_data[primary_day]['heatmap'][idx_safe]
        
        info = {
            "portfolio_value": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "reward": reward,
            "price": current_price,
            "date": self.raw_df.iloc[self.current_step]['date'], # Needed for Backtest Chart
            "rsi": cur_rsi,
            "stoch_rsi": cur_stoch,
            "vp_heatmap": heatmap
        }

        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step}: NetWorth={self.net_worth:.2f}")

        return obs, reward, terminated, truncated, info