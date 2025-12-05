import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
from scipy.signal import argrelextrema

# Delegating heavy lifting to volume_profile.py
from volume_profile import get_rolling_vp

class EnhancedTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=50, vp_days=None, vp_bins=40,
                 buy_threshold=0.5, sell_threshold=-0.5, precalculated_vp=None, trading_fee_multiplier=0.00075, phase=1):
        super(EnhancedTradingEnv, self).__init__()
        
        # --- CONFIGURATION ---
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        # Default to [3, 7] if None
        self.vp_days = vp_days if vp_days else [3, 7]
        self.vp_bins = vp_bins
        self.trading_fee_multiplier = trading_fee_multiplier
        
        # --- THRESHOLDS ---
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # --- 1. DATA PREP ---
        self.raw_df = df.reset_index(drop=False)
        self.df = self.raw_df.copy()
        
        if precalculated_vp:
            self.vp_data = precalculated_vp
        else:
            # Fallback to calculating it (only if not passed)
            self.vp_data = {}
            print(f"--- Initializing EnhancedTradingEnv (Target Bins: {self.vp_bins}) ---")
            
            for days in self.vp_days:
                # All caching/hashing logic is now in volume_profile.py
                self.vp_data[days] = get_rolling_vp(self.raw_df, days, bins=self.vp_bins)
        
        # Standard Features
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)

        # --- FIX: Rolling Max Normalization ---
        # Instead of Z-score or simple Log, we scale volume relative to the last 100 steps.
        # This ensures the value is almost always between 0.0 and 1.0

        # 1. Get Rolling Max Volume (Window = 100 or similar to your observation window)
        vol_rolling_max = self.df['volume'].rolling(window=100, min_periods=1).max()

        # 2. Divide current volume by rolling max (Safe division)
        self.df['volume_norm'] = self.df['volume'] / (vol_rolling_max + 1e-8)

        # 3. Fill NaNs just in case
        self.df['volume_norm'] = self.df['volume_norm'].fillna(0)

        # Indicators
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi_norm'] = self.df['rsi'] / 100.0

        min_rsi = self.df['rsi'].rolling(window=14).min()
        max_rsi = self.df['rsi'].rolling(window=14).max()
        div = max_rsi - min_rsi
        div[div == 0] = 1e-8
        self.df['stoch_rsi'] = (self.df['rsi'] - min_rsi) / div
        raw_stoch = self.df['stoch_rsi']
        self.df['stoch_rsi_norm'] = raw_stoch
        mask = pd.isna(raw_stoch) | ((self.df.index < 20) & (raw_stoch == 0))
        self.df.loc[mask, 'stoch_rsi_norm'] = 0.5
        self.df['stoch_rsi_norm'] = np.clip(self.df['stoch_rsi_norm'], 0.0, 1.0)

        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_norm'] = (self.df['macd'] / self.df['close']) * 100
        self.df['macd_sig_norm'] = (self.df['macd_signal'] / self.df['close']) * 100

        # EMA 50 for trend
        self.df['ema_50'] = self.df['close'].ewm(span=50, adjust=False).mean()

        # --- UPDATED: Z-Score Trend Scaling ---
        raw_trend = (self.df['close'] - self.df['ema_50']) / self.df['ema_50']
        std_roll = self.df['close'].rolling(100).std().fillna(1)  # Avoid div0
        self.df['trend_ema50'] = (raw_trend / std_roll) * 0.5
        self.df['trend_ema50'] = np.clip(self.df['trend_ema50'], -1, 1)

        # --- NEW: ATR & Regime Features ---
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['atr'] = tr.rolling(14).mean().fillna(0)
        self.df['atr_norm'] = self.df['atr'] / self.df['close']

        self.df['regime'] = self.df['trend_ema50'] / (self.df['atr_norm'] + 1e-8)
        self.df['regime'] = np.clip(self.df['regime'], -2, 2)

        # Stochastic 14
        min_low = self.df['low'].rolling(window=14).min()
        max_high = self.df['high'].rolling(window=14).max()
        self.df['stoch_14'] = (self.df['close'] - min_low) / (max_high - min_low + 1e-8)
        self.df['stoch_14'] = self.df['stoch_14'].fillna(0.5)
        self.df['stoch_14'] = np.clip(self.df['stoch_14'], 0.0, 1.0)

        self.features = ['close_pct', 'volume_norm', 'rsi_norm', 'stoch_rsi_norm', 'macd_norm', 'macd_sig_norm', 'trend_ema50', 'atr_norm', 'regime']
        self.divergence_window = 40
        self.div_features = [
            'bull_div_stoch9',
            'bear_div_stoch9',
            'bull_div_stoch14',
            'bear_div_stoch14',
            'bull_div_rsi',
            'bear_div_rsi'
        ]
        self.div_scores = {k: 0.0 for k in self.div_features}  # Initialize state
        self.df.fillna(0, inplace=True)
        
        self.market_features = self.df[self.features].values.astype(np.float32)
        self.raw_prices = self.df['close'].values.astype(np.float32)

        # --- FIX: Initialize feature names explicitly for Main.py ---
        self.feature_names = self.get_feature_names()

        # --- 3. SPACE DEFINITION (Updated for new features) ---
        market_obs_size = self.lookback_window * len(self.features)
        account_obs_size = 2
        vp_obs_size = len(self.vp_days) * (3 + self.vp_bins)
        total_obs_size = market_obs_size + account_obs_size + vp_obs_size + len(self.div_features)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32)

        self.stability_penalty_coef = 0.01
        self.interaction_penalty = 0.0
        self.reward_scaling = 1.0

        # --- DYNAMIC CHURN SETTINGS ---
        # Target: Agent should ideally hold for 24 steps (e.g. 1 day if hourly)
        # If it flips sooner, it pays a penalty.
        self.target_hold_duration = 24
        self.churn_penalty_max = 0.5

        self.prev_actions = deque(maxlen=3)

        self.max_lookback = max(max([d * 24 for d in self.vp_days]), 30) + self.lookback_window

        self.phase = phase
        self.prev_sign = 0  # Initialize for switch penalty
        self.reset()

    def _detect_divergences(self, series, price_series, window=40, tolerance=8):
        if self.current_step < window + 20:
            return 0.0, 0.0

        start_idx = self.current_step - window
        recent_prices = price_series[start_idx: self.current_step + 1]
        recent_series = series[start_idx: self.current_step + 1]

        # Find swing lows and highs in price
        price_lows_idx = argrelextrema(recent_prices, np.less, order=3)[0]
        price_highs_idx = argrelextrema(recent_prices, np.greater, order=3)[0]

        bullish = bearish = 0.0

        # Helper to check if swing is recent enough (within last `tolerance` bars from end)
        is_recent = lambda idx: (len(recent_prices) - 1 - idx) <= tolerance

        # Bullish divergence: lower low in price, higher low in oscillator
        if len(price_lows_idx) >= 2:
            p2 = price_lows_idx[-1]   # most recent low
            p1 = price_lows_idx[-2]   # previous low

            if is_recent(p2) and recent_prices[p2] < recent_prices[p1]:
                # Find oscillator lows near price lows
                osc_lows_idx = argrelextrema(recent_series, np.less, order=3)[0]
                near_p1 = [i for i in osc_lows_idx if abs(i - p1) <= 8]
                near_p2 = [i for i in osc_lows_idx if abs(i - p2) <= 8]
                if near_p1 and near_p2:
                    if recent_series[near_p2[0]] > recent_series[near_p1[0]]:
                        bullish = 1.0

        # Bearish divergence: higher high in price, lower high in oscillator
        if len(price_highs_idx) >= 2 and bullish == 0.0:  # avoid double signal
            p2 = price_highs_idx[-1]
            p1 = price_highs_idx[-2]

            if is_recent(p2) and recent_prices[p2] > recent_prices[p1]:
                osc_highs_idx = argrelextrema(recent_series, np.greater, order=3)[0]
                near_p1 = [i for i in osc_highs_idx if abs(i - p1) <= 8]
                near_p2 = [i for i in osc_highs_idx if abs(i - p2) <= 8]
                if near_p1 and near_p2:
                    if recent_series[near_p2[0]] < recent_series[near_p1[0]]:
                        bearish = 1.0

        return bullish, bearish

    def get_feature_names(self):
        """Returns a list of labels for the observation space."""
        # This must match exactly the order of np.concatenate in _next_observation
        names = [
            "volume_norm", "trend_ema", "close_pct",
            "rsi_norm", "stoch_rsi", "macd_norm", "macd_sig", "atr_norm",
            "regime", "heatmap_max"
        ]
        # Add Heatmap buckets
        names += [f"vp_bucket_{i}" for i in range(self.vp_bins)]
        # Add Divergence features
        names += self.div_features

        return names

    def _take_action(self, action):
        action_val = float(action[0])
        current_price = self.raw_prices[self.current_step]

        # --- TRADE LOGIC ---
        if abs(action_val - self.prev_action) > 0.1:
            self.trade_count += 1
            self.prev_action = action_val

        # --- UPDATED: Dynamic Thresholds via ATR ---
        atr_idx = self.features.index('atr_norm')
        atr_norm = self.market_features[self.current_step, atr_idx]
        thresh_mult = 1 + (atr_norm * 0.5)  # Higher vol → higher threshold (needs stronger action)
        
        dynamic_buy_threshold = self.buy_threshold * thresh_mult
        dynamic_sell_threshold = self.sell_threshold * thresh_mult  # More negative in high vol

        # --- CONFIGURABLE THRESHOLDS ---
        if action_val > dynamic_buy_threshold: # Buy
            amount_to_invest = self.balance * action_val
            if amount_to_invest > 10:
                shares_bought = amount_to_invest / current_price
                fee = amount_to_invest * self.trading_fee_multiplier
                self.balance -= amount_to_invest + fee
                self.shares_held += shares_bought
        
        elif action_val < dynamic_sell_threshold: # Sell
            shares_to_sell = self.shares_held * abs(action_val)
            if shares_to_sell * current_price > 10:
                trade_value = shares_to_sell * current_price
                fee = trade_value * self.trading_fee_multiplier
                self.balance += trade_value - fee
                self.shares_held -= shares_to_sell

        self.net_worth = self.balance + (self.shares_held * current_price)
        self.history_net_worth.append(self.net_worth)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def _calculate_reward(self, action, trade_occurred=False):
        step_reward = 0.0

        # 1. Returns
        pct_change = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        step_reward += pct_change * 100

        # 2. Stability Penalty
        if len(self.prev_actions) > 1:
            prev_act = self.prev_actions[-2]
            step_reward -= abs(action - prev_act) * self.stability_penalty_coef

        # 3. Dynamic Churn Penalty
        if trade_occurred:
            duration = self.steps_since_last_trade
            if duration < self.target_hold_duration:
                penalty_factor = 1.0 - (duration / self.target_hold_duration)
                churn_cost = self.churn_penalty_max * penalty_factor
                step_reward -= churn_cost

        return step_reward

    def set_phase(self, new_phase):
        self.phase = new_phase

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0
        self.prev_shares_held = 0
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.prev_action = 0
        self.prev_sign = 0  # Reset for switch penalty
        self.history_net_worth = [self.initial_balance]
        self.episode_returns = deque(maxlen=24)  # For Sortino (last 24 hours)
        
        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback

        self.steps_since_last_trade = 0
        self.prev_actions = deque(maxlen=3)

        return self._next_observation(), {}

    def _next_observation(self):
        window_start = self.current_step - self.lookback_window
        std_features = self.market_features[window_start : self.current_step].flatten()
        
        current_price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * current_price) / self.initial_balance
        account_features = np.array([balance_norm, holdings_norm], dtype=np.float32)
        
        vp_features_list = []
        for days in self.vp_days:
            data = self.vp_data[days]
            poc = data['poc'][self.current_step]
            vah = data['vah'][self.current_step]
            val = data['val'][self.current_step]
            heatmap = data['heatmap'][self.current_step].astype(np.float32)

            # --- FIX: Normalize Heatmap ---
            # Normalize by SUM (Probability Distribution)
            # This ensures the total signal strength of the heatmap is exactly 1.0.
            # It puts the heatmap on equal footing with RSI/EMA.
            total_volume = np.sum(heatmap)
            if total_volume > 0:
                heatmap = heatmap / total_volume
            else:
                heatmap = np.zeros_like(heatmap)

            if current_price > 0 and poc > 0:
                dist_poc = (poc - current_price) / current_price
                dist_vah = (vah - current_price) / current_price
                dist_val = (val - current_price) / current_price
            else:
                dist_poc, dist_vah, dist_val = 0, 0, 0

            vp_features_list.extend([dist_poc, dist_vah, dist_val])
            vp_features_list.extend(heatmap)
            
        vp_features = np.array(vp_features_list, dtype=np.float32)

        # Detect divergences (only when current_step is valid)
        bull9, bear9 = self._detect_divergences(
            self.df['stoch_rsi_norm'].values,
            self.raw_prices,
            window=40, tolerance=8
        )
        bull14, bear14 = self._detect_divergences(
            self.df['stoch_14'].values,
            self.raw_prices, window=40, tolerance=8
        )
        bull_rsi, bear_rsi = self._detect_divergences(
            self.df['rsi_norm'].values,
            self.raw_prices, window=40, tolerance=8
        )

        # Decay the Signal
        decay_rate = 0.95

        # 1. Decay existing scores
        for k in self.div_scores:
            self.div_scores[k] *= decay_rate

        # 2. Add new detections (if any)
        if bull9 > 0: self.div_scores['bull_div_stoch9'] = 1.0
        if bear9 > 0: self.div_scores['bear_div_stoch9'] = 1.0
        if bull14 > 0: self.div_scores['bull_div_stoch14'] = 1.0
        if bear14 > 0: self.div_scores['bear_div_stoch14'] = 1.0
        if bull_rsi > 0: self.div_scores['bull_div_rsi'] = 1.0
        if bear_rsi > 0: self.div_scores['bear_div_rsi'] = 1.0

        # 3. Create vector from self.div_scores, NOT the raw detection variables
        div_vector = np.array([self.div_scores[k] for k in self.div_features], dtype=np.float32)

        full_obs = np.concatenate((std_features, account_features, vp_features, div_vector))

        # --- DEBUG LOGGING ---
        # Print stats every 10000 steps to avoid spamming, but see what's happening
        if self.current_step % 10000 == 0:
            print(f"\n[DEBUG Step {self.current_step}] Feature Magnitudes:")

            # 1. Check Volume Magnitude
            vol_val = self.df.iloc[self.current_step]['volume_norm']
            print(f"  > Volume Norm Input:   {vol_val:.5f}  (Should be 0.0 - 1.0)")

            # 2. Check Trend Magnitude (The likely culprit)
            trend_val = self.df.iloc[self.current_step]['trend_ema50']
            print(f"  > Trend EMA Input:     {trend_val:.5f}  (Now z-scored and clipped -1 to 1)")

            # Debug suggestion:
            current_close = current_price
            ema_val = self.df.iloc[self.current_step]['ema_50']
            print(f"DEBUG EMA: Close={current_close}, EMA={ema_val}, Diff={current_close - ema_val}")

            # 3. Check Close Pct
            close_pct_val = self.df.iloc[self.current_step]['close_pct']
            print(f"  > Close Pct Input:     {close_pct_val:.5f}")

            # 4. Check RSI Norm
            rsi_norm_val = self.df.iloc[self.current_step]['rsi_norm']
            print(f"  > RSI Norm Input:      {rsi_norm_val:.5f}")

            # 5. Check Stoch RSI
            stoch = self.df.iloc[self.current_step]['stoch_rsi_norm']
            print(f"  > Stoch RSI Input:     {stoch:.5f}")

            # 6. Check MACD Norm
            macd_norm_val = self.df.iloc[self.current_step]['macd_norm']
            print(f"  > MACD Norm Input:     {macd_norm_val:.5f}")

            # 7. Check MACD Sig Norm
            macd_sig_norm_val = self.df.iloc[self.current_step]['macd_sig_norm']
            print(f"  > MACD Sig Norm Input: {macd_sig_norm_val:.5f}")

            # 8. Check ATR Norm
            atr_norm_val = self.df.iloc[self.current_step]['atr_norm']
            print(f"  > ATR Norm Input:      {atr_norm_val:.5f}")

            # 9. Check Regime
            regime_val = self.df.iloc[self.current_step]['regime']
            print(f"  > Regime Input:        {regime_val:.5f} (-2 to 2)")

            # 10. Check VP Heatmap Magnitude
            # Access the first available VP day key
            first_day = self.vp_days[0]
            vp_sample = self.vp_data[first_day]['heatmap'][self.current_step]
            print(f"  > VP Heatmap Max:      {np.max(vp_sample):.2f} (Now normalized by sum, max <=1.0)")
            print(f"  > VP Heatmap Values:   {vp_sample}")
            
            print(f"  > Bull Div Stoch9:    {div_vector[0]:.5f}")
            print(f"  > Bear Div Stoch9:    {div_vector[1]:.5f}")
            print(f"  > Bull Div Stoch14:   {div_vector[2]:.5f}")
            print(f"  > Bear Div Stoch14:   {div_vector[3]:.5f}")
            print(f"  > Bull Div RSI:       {div_vector[4]:.5f}")
            print(f"  > Bear Div RSI:       {div_vector[5]:.5f}")
                        
        return full_obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])

        # 1. Detect if a Trade (Sign Flip) is happening
        prev_action = self.prev_actions[-1] if len(self.prev_actions) > 0 else 0.0
        trade_occurred = False
        if abs(action_val) > 0.1:
            if np.sign(action_val) != np.sign(prev_action):
                trade_occurred = True

        # 2. Execute action
        self._take_action(action)

        # 3. Calculate reward
        reward = self._calculate_reward(action_val, trade_occurred)

        # 4. Update Duration Counter
        if trade_occurred:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1

        # Append to prev_actions
        self.prev_actions.append(action_val)

        self.prev_net_worth = self.net_worth
        self.prev_shares_held = self.shares_held

        terminated = False
        truncated = False
        if self.current_step >= len(self.df) - 1: truncated = True
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            reward = -10

        obs = self._next_observation()

        # Info
        primary_day = self.vp_days[0]
        idx_safe = min(self.current_step, len(self.vp_data[primary_day]['heatmap']) - 1)
        heatmap = self.vp_data[primary_day]['heatmap'][idx_safe]

        info = {
            "portfolio_value": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "reward": reward,
            "price": current_price,
            "current_price": current_price,
            "ema50": self.df.iloc[self.current_step].get('ema_50', 0),
            "timestamp": self.raw_df.iloc[self.current_step]['timestamp'],
            "vp_heatmap": heatmap
        }

        return obs, reward, terminated, truncated, info