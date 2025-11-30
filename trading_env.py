import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Import your helper
from volume_profile import get_rolling_vp
from features import get_features

class TradingEnv(gym.Env):
    total_steps = 0  # Global step counter across all episodes

    def __init__(self, df, vp7_df=None, vp30_df=None, initial_balance=10000, lookback_window=30, vp_days=None, buy_threshold=0.1, sell_threshold=-0.1):
        super(TradingEnv, self).__init__()

        self.trade_history = []
        self.returns = []  # Track returns for Sortino calculation

        # --- DATA PREP ---
        # Keep 'date' as a column so we can log it later
        self.raw_df = df.reset_index(drop=False)
        print(f"DEBUG: raw_df columns after reset_index: {list(self.raw_df.columns)}")
        # Rename 'timestamp' to 'date' if present
        if 'timestamp' in self.raw_df.columns:
            self.raw_df.rename(columns={'timestamp': 'date'}, inplace=True)
            print("DEBUG: Renamed 'timestamp' to 'date'")
        self.df = self.raw_df.copy()
        print(f"DEBUG: Attempting to set index on column: 'date', available columns: {list(self.df.columns)}")
        self.df.set_index('date', inplace=True)
        
        # 1. Standard Features
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)
        self.df['volume_norm'] = (self.df['volume'] - self.df['volume'].mean()) / (self.df['volume'].std() + 1e-8)

        # 2. TECHNICAL INDICATORS
        # A. RSI (14)
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi_norm'] = self.df['rsi'] / 100.0

        # B. Stochastic RSI (14)
        min_rsi = self.df['rsi'].rolling(window=14).min()
        max_rsi = self.df['rsi'].rolling(window=14).max()
        self.df['stoch_rsi'] = (self.df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-8)
        self.df['stoch_rsi_norm'] = self.df['stoch_rsi'].fillna(0.5)

        # C. MACD (12, 26, 9)
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        
        # Normalize MACD
        self.df['macd_norm'] = self.df['macd'] / self.df['close']
        self.df['macd_sig_norm'] = self.df['macd_signal'] / self.df['close']

        # --------------------------------------------------------------------

        self.df.fillna(0, inplace=True)

        self.raw_prices = self.df['close'].values.astype(np.float32)

        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.phase = 1  # Start at Phase 1 (Simple PnL)

        # --- VOLUME PROFILE ---
        self.vp_days = vp_days if vp_days else [7, 30]

        if vp7_df is not None and vp30_df is not None:
            self.vp7_df = vp7_df.set_index(self.df.index)
            self.vp30_df = vp30_df.set_index(self.df.index)
            print("--- Using provided VP DataFrames ---")
        else:
            self.vp_data = {}
            print(f"--- Initializing Environment (VP Days: {self.vp_days}) ---")

            for days in self.vp_days:
                vp_result = get_rolling_vp(self.raw_df, days, bin_percent=0.005)
                self.vp_data[days] = vp_result

            # Create DataFrames for features.py
            vp7_dict = self.vp_data[7]
            self.vp7_df = pd.DataFrame({
                'poc': vp7_dict['poc'],
                'vah': vp7_dict['vah'],
                'val': vp7_dict['val'],
                'hvn': vp7_dict['hvn'],
                'lvn': vp7_dict['lvn'],
                'heatmap': [row for row in vp7_dict['heatmap']]
            }, index=self.df.index)

            vp30_dict = self.vp_data[30]
            self.vp30_df = pd.DataFrame({
                'poc': vp30_dict['poc'],
                'vah': vp30_dict['vah'],
                'val': vp30_dict['val'],
                'hvn': vp30_dict['hvn'],
                'lvn': vp30_dict['lvn'],
                'heatmap': [row for row in vp30_dict['heatmap']]
            }, index=self.df.index)
            
        # Define Observation Space
        features_obs_size = 259  # from get_features
        account_obs_size = 2
        total_obs_size = features_obs_size + account_obs_size
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )
        
        # Adjust lookback to account for MACD warmup
        self.max_lookback = max(max([d * 24 for d in self.vp_days]), 30) + self.lookback_window

    def set_phase(self, phase_number):
        """Called by main.py to upgrade the level"""
        self.phase = phase_number
        print(f"⚠️ Environment upgraded to PHASE {self.phase}")

    def get_current_phase(self):
        """Determine current learning phase based on total training steps"""
        steps = TradingEnv.total_steps
        if steps < 1000000:  # Phase 1: 0-1M steps
            return 1
        elif steps < 2000000:  # Phase 2: 1M-2M
            return 2
        elif steps < 3000000:  # Phase 3: 2M-3M
            return 3
        elif steps < 4000000:  # Phase 4: 3M-4M
            return 4
        else:  # Phase 5: 4M+
            return 5
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0

        # --- PHASE 3: Track High Water Mark for Drawdown Penalty ---
        self.max_net_worth = self.initial_balance

        # Reset returns for Sortino tracking
        self.returns = []
        
        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback

        return self._next_observation(), {}

    def _next_observation(self):
        # 1. Features from get_features
        t = self.df.index[self.current_step]
        features = get_features(self.df, self.vp7_df, self.vp30_df, t)

        # 2. Account Data
        current_price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * current_price) / self.initial_balance
        account_features = np.array([balance_norm, holdings_norm], dtype=np.float32)

        # Combine
        full_obs = np.concatenate((features, account_features))
        return full_obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        TradingEnv.total_steps += 1
        # current_phase = self.get_current_phase()

        current_price = self.raw_prices[self.current_step]
        action_val = action[0]
        dynamic_buy_threshold = abs(action[1])
        dynamic_sell_threshold = -abs(action[2])

        # Validation to prevent invalid thresholds
        if dynamic_buy_threshold <= 0:
            dynamic_buy_threshold = 0.01
        if dynamic_sell_threshold >= 0:
            dynamic_sell_threshold = -0.01

        # DEBUG: Log action and trades
        # print(f"DEBUG STEP: action_val={action_val:.4f}, balance={self.balance:.2f}, shares={self.shares_held:.6f}, net_worth={self.net_worth:.2f}")

        # --- TRADE LOGIC (Higher Fee 0.15% to prevent overfitting) ---
        REALISTIC_FEE = 0.0015
        trade_penalty = 0
        trade_happened = False

        if action_val > dynamic_buy_threshold: # Buy
            amount_to_invest = self.balance * action_val
            if amount_to_invest > 10:
                shares_bought = amount_to_invest / current_price
                self.balance -= amount_to_invest
                self.shares_held += shares_bought
                trade_penalty = REALISTIC_FEE
                trade_happened = True
                self.trade_history.append({'step': self.current_step, 'type': 'buy', 'shares': shares_bought, 'price': current_price})
                # print(f"DEBUG: BUY {shares_bought:.6f} shares at {current_price:.2f}")
            else:
                trade_penalty = 0.01
        elif action_val < dynamic_sell_threshold: # Sell
            shares_to_sell = self.shares_held * abs(action_val)
            if shares_to_sell * current_price > 10:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                trade_penalty = REALISTIC_FEE
                trade_happened = True
                self.trade_history.append({'step': self.current_step, 'type': 'sell', 'shares': shares_to_sell, 'price': current_price})
                # print(f"DEBUG: SELL {shares_to_sell:.6f} shares at {current_price:.2f}")
            else:
                trade_penalty = 0.01


        self.net_worth = self.balance + (self.shares_held * current_price)
        # print(f"DEBUG: Updated net_worth={self.net_worth:.2f}")

        # Track returns for Sortino
        step_reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        self.returns.append(step_reward)

        # --- PHASE-BASED REWARD CALCULATION ---

        # 1. Update High Water Mark (ATH)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # 2. Base PnL Reward
        reward = step_reward * 100

        # 3. Phase-specific adjustments
        reward -= trade_penalty
        if self.phase >= 2:
            if step_reward < 0:
                reward -= abs(step_reward) * 5.0
        if self.phase >= 3:
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            reward -= (drawdown * 0.1)

        # Phase 4+: Volume Profile Alignment Bonus
        if self.phase >= 4:
            vp_bonus = self._calculate_vp_bonus(current_price, trade_happened, action_val > 0 if trade_happened else None)
            reward += vp_bonus
        
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
        t = self.df.index[self.current_step]

        # EXTRACT RAW VALUES FOR WANDB
        raw_poc = self.vp7_df.loc[t, 'poc'] if primary_day == 7 else self.vp30_df.loc[t, 'poc']
        raw_vah = self.vp7_df.loc[t, 'vah'] if primary_day == 7 else self.vp30_df.loc[t, 'vah']
        raw_val = self.vp7_df.loc[t, 'val'] if primary_day == 7 else self.vp30_df.loc[t, 'val']

        # EXTRACT INDICATORS
        cur_rsi = self.df.iloc[self.current_step]['rsi']
        cur_stoch = self.df.iloc[self.current_step]['stoch_rsi']
        cur_macd = self.df.iloc[self.current_step]['macd']
        cur_macd_sig = self.df.iloc[self.current_step]['macd_signal']

        info = {
            "portfolio_value": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "buy_threshold": dynamic_buy_threshold,
            "sell_threshold": dynamic_sell_threshold,
            "reward": reward,
            "price": current_price,
            "max_net_worth": self.max_net_worth, # Log ATH

            # Send raw values for charts
            "poc": raw_poc,
            "vah": raw_vah,
            "val": raw_val,

            "rsi": cur_rsi,
            "stoch_rsi": cur_stoch,
            "macd": cur_macd,
            "macd_sig": cur_macd_sig,

            "last_10_trades": self.trade_history[-10:],
            "current_phase": self.phase
        }

        # Console Log with Date
        if self.current_step % 100 == 0:
            current_date = self.raw_df.iloc[self.current_step]['date']
            dist_pct = ((current_price - raw_poc) / raw_poc) * 100 if raw_poc != 0 else 0

            print(f"Step {self.current_step} [{current_date}]: Price={current_price:.0f} | POC={raw_poc:.0f} ({dist_pct:+.2f}%) | NetWorth={self.net_worth:.0f} | MaxNetWorth={self.max_net_worth:.0f}")

        return obs, reward, terminated, truncated, info

    def _calculate_vp_bonus(self, current_price, trade_happened, is_buy):
        """Calculate Volume Profile alignment bonus for Phase 4+"""
        bonus = 0.0
        λ1, λ2, λ3 = 0.1, 0.1, 0.1  # Weights

        t = self.df.index[self.current_step]
        poc = self.vp7_df.loc[t, 'poc']
        hvn = self.vp7_df.loc[t, 'hvn'] if isinstance(self.vp7_df.loc[t, 'hvn'], list) else []
        lvn = self.vp7_df.loc[t, 'lvn'] if isinstance(self.vp7_df.loc[t, 'lvn'], list) else []

        threshold = 0.01  # 1% proximity

        if trade_happened:
            if is_buy:
                # Entry near LVN fade (buying low)
                if any(abs(current_price - lv) / current_price < threshold for lv in lvn):
                    bonus += λ1
                # Fighting POC: buying above POC
                if current_price > poc:
                    bonus -= λ3
            else:
                # Entry near HVN fade (selling high)
                if any(abs(current_price - hv) / current_price < threshold for hv in hvn):
                    bonus += λ1
                # Exit near HVN
                if any(abs(current_price - hv) / current_price < threshold for hv in hvn):
                    bonus += λ2
                # Fighting POC: selling below POC
                if current_price < poc:
                    bonus -= λ3

        return bonus