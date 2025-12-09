import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# Delegating heavy lifting to volume_profile.py
from volume_profile import get_rolling_vp
import logging

class EnhancedTradingEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'render_fps': 4,
    }

    def __init__(self, df, initial_balance=10000, lookback_window=50, vp_days=None, vp_bins=40,
                      buy_threshold=0.5, sell_threshold=-0.5, precalculated_vp=None, trading_fee_multiplier=0.00075, phase=1, min_trade_value_usd=10.0,
                      pair='BTCUSDT', timeframe='1h', split_date=None):
        logging.info(f"--- Initializing EnhancedTradingEnv (Target Bins: {vp_bins}) ---")
        super(EnhancedTradingEnv, self).__init__()

        # Update metadata with instance-specific info
        self.metadata.update({
            'pair': pair,
            'timeframe': timeframe,
            'data_range': f"{df.index[0]} to {df.index[-1]}" if not df.empty else None,
            'split_date': split_date,
        })

        # === OVERTRADING FIXES ===
        self.transaction_cost_rate = 0.0015      # 0.15% per trade (Binance spot taker fee ≈ 0.1% + slippage)
        self.reward_fee_multiplier = 10.0        # Magnify fee 10x in reward calculation to stop churning
        self.action_penalty = 0.0005             # tiny L1 penalty to discourage twitching
        self.last_trade_cost = 0
        
        # --- CONFIGURATION ---
        self.initial_balance = initial_balance
        self.last_price = 0
        self.current_position = 0
        self.prev_action = 0.0  # NEW: Track previous action for inertia
        # REMOVED: Look-ahead horizons (no more exploit)
        self.lookback_window = lookback_window
        # Default to [3, 7] if None
        self.vp_days = vp_days if vp_days else [3, 7]
        self.vp_bins = vp_bins
        self.trading_fee_multiplier = trading_fee_multiplier
        
        # === FIX: Use args directly so 0.0 works ===
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # --- 1. DATA PREP ---
        self.raw_df = df.reset_index(drop=False)
        self.df = self.raw_df.copy()

        # Drop non-numeric columns to avoid conversion errors
        self.df = self.df.drop(columns=['timestamp'], errors='ignore')

        if precalculated_vp:
            self.vp_data = precalculated_vp
        else:
            # Fallback to calculating it (only if not passed)
            self.vp_data = {}
            logging.info(f"--- Initializing EnhancedTradingEnv (Target Bins: {self.vp_bins}) ---")
            
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
        # RSI & Stoch RSI: Center them!
        # Old: 0 to 100 (or 0 to 1). Neural Net sees 0.5 as "Positive Activation".
        # New: -1.0 to 1.0. Neural Net sees 0.0 as "Neutral".
        # Formula: (Value - 50) / 50
        self.df['rsi_norm'] = (self.df['rsi'] - 50.0) / 50.0

        min_rsi = self.df['rsi'].rolling(window=14).min()
        max_rsi = self.df['rsi'].rolling(window=14).max()
        div = max_rsi - min_rsi
        div[div == 0] = 1e-8
        self.df['stoch_rsi'] = (self.df['rsi'] - min_rsi) / div
        raw_stoch = self.df['stoch_rsi']
        # Assuming stoch_rsi is 0-1. If it's 0-100, use (x - 50.0) / 50.0
        # Let's handle 0-1 standard case
        self.df['stoch_rsi_norm'] = (raw_stoch - 0.5) / 0.5
        mask = pd.isna(raw_stoch) | ((self.df.index < 20) & (raw_stoch == 0))
        self.df.loc[mask, 'stoch_rsi_norm'] = 0.0  # Neutral for centered

        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()

        # MACD: Dynamic Expanding Normalization
        # MACD is unbounded. Dividing by Price (old way) is weak.
        # Dividing by "Max MACD Seen So Far" is robust.
        macd_abs_max = self.df['macd'].abs().expanding(min_periods=200).max()
        macd_abs_max = macd_abs_max.replace(0, 1.0).fillna(1.0)  # Safety
        self.df['macd_norm'] = self.df['macd'] / macd_abs_max

        sig_abs_max = self.df['macd_signal'].abs().expanding(min_periods=200).max()
        sig_abs_max = sig_abs_max.replace(0, 1.0).fillna(1.0)
        self.df['macd_sig_norm'] = self.df['macd_signal'] / sig_abs_max

        # EMA 50 for trend
        self.df['ema_50'] = self.df['close'].ewm(span=50, adjust=False).mean()

        # --- NEW: ATR & Regime Features ---
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['atr'] = tr.rolling(14).mean().fillna(0.01)
        self.df['atr_norm'] = self.df['atr'] / self.df['close']

        # --- ROBUST TREND EMA CALCULATION ---
        # Calculate Distance (Raw Deviation)
        dist = self.df['close'] - self.df['ema_50']

        # Calculate Denominator: Dynamic Expanding Max
        # "What is the largest deviation we have seen SO FAR?"
        # We use an expanding window so it learns and adapts over time.
        # min_periods=200 ensures we have some history before trusting the max.
        max_dist_so_far = dist.abs().expanding(min_periods=200).max()

        # Handle the startup period (first 200 candles) safely
        # We fill NaNs with the first valid max, or a fallback (1% of price)
        fallback_denom = self.df['close'] * 0.01
        max_dist_so_far = max_dist_so_far.fillna(fallback_denom)

        # Safety: Avoid division by zero
        max_dist_so_far = max_dist_so_far.replace(0, 1.0)

        # Normalize
        # Value will be exactly 1.0 or -1.0 when a new "Record Trend" is set.
        # Otherwise, it floats nicely between -0.8 and 0.8.
        self.df['trend_ema_norm'] = dist / max_dist_so_far

        self.df['regime'] = self.df['trend_ema_norm'] / (self.df['atr_norm'] + 1e-8)
        self.df['regime'] = np.clip(self.df['regime'], -2, 2)

        # Stochastic 14
        min_low = self.df['low'].rolling(window=14).min()
        max_high = self.df['high'].rolling(window=14).max()
        self.df['stoch_14'] = (self.df['close'] - min_low) / (max_high - min_low + 1e-8)
        self.df['stoch_14'] = self.df['stoch_14'].fillna(0.5)
        self.df['stoch_14'] = np.clip(self.df['stoch_14'], 0.0, 1.0)

        self.features = ['close_pct', 'volume_norm', 'rsi_norm', 'stoch_rsi_norm', 'macd_norm', 'macd_sig_norm', 'trend_ema_norm', 'atr_norm', 'regime']
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

        # Convert the entire dataframe to a float32 numpy matrix for speed
        self.data_matrix = self.df.values.astype(np.float32)

        # Column indices for fast access
        self.atr_norm_idx = self.df.columns.get_loc('atr_norm')
        self.close_pct_idx = self.df.columns.get_loc('close_pct')
        self.ema_50_idx = self.df.columns.get_loc('ema_50')
        self.volume_norm_idx = self.df.columns.get_loc('volume_norm')
        self.trend_ema_norm_idx = self.df.columns.get_loc('trend_ema_norm')
        self.rsi_norm_idx = self.df.columns.get_loc('rsi_norm')
        self.stoch_rsi_norm_idx = self.df.columns.get_loc('stoch_rsi_norm')
        self.macd_norm_idx = self.df.columns.get_loc('macd_norm')
        self.macd_sig_norm_idx = self.df.columns.get_loc('macd_sig_norm')
        self.regime_idx = self.df.columns.get_loc('regime')
        self.stoch_14_idx = self.df.columns.get_loc('stoch_14')

        # Use data_matrix for market_features to optimize
        feature_indices = [self.df.columns.get_loc(f) for f in self.features]
        self.market_features = self.data_matrix[:, feature_indices]
        self.raw_prices = self.data_matrix[:, self.df.columns.get_loc('close')]

        # --- FIX: Initialize feature names explicitly for Main.py ---
        self.feature_names = self.get_feature_names()

        # --- 3. SPACE DEFINITION (Updated for new features) ---
        market_obs_size = self.lookback_window * len(self.features)
        account_obs_size = 2
        vp_obs_size = len(self.vp_days) * (3 + self.vp_bins)
        recurrent_obs_size = 10  # 5 recent actions + 5 position deltas
        total_obs_size = (
            market_obs_size
            + account_obs_size
            + vp_obs_size
            + len(self.div_features)
            + recurrent_obs_size
        )

        # Action now encodes signed target exposure in [-1, 1]
        # -1 = max short, 0 = flat, 1 = max long (before leverage scaling)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32)

        self.stability_penalty_coef = 0.02   # Slight increase (0.01 -> 0.02)
        self.interaction_penalty = 0.0
        self.entry_price = 0.0  # + NEW: Initialize to fix AttributeError in step
        self.reward_scaling = 1.0

        # === FIX: Soften churn/cooldown to not over-punish trading ===
        self.target_hold_duration = 24       # Target 24h hold
        self.cooldown_steps = 12             # Force minimum 12-step (12h) wait between flipping sides

        # REWARD PENALTIES
        # We switch to a more conservative approach

        # State tracking
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.shares_held = 0.0
        self.prev_net_worth = initial_balance
        self.prev_shares_held = 0.0
        self.current_step = self.lookback_window
        self.trades_in_episode = 0
        self.steps_since_last_trade = 0
        self.has_traded_once = False
        self.max_net_worth = initial_balance
        self.portfolio_returns = deque(maxlen=100)
        self.returns = deque(maxlen=100)
        self.recent_actions = deque(maxlen=5)
        self.recent_position_deltas = deque(maxlen=5)
        self.action_history = []
        self.history_net_worth = [initial_balance]

        # Leverage limits (conservative for now)
        self.max_leverage = 2.0  # Reduced from 5.0 to prevent blowups
        self.leverage_buffer = 0.95

        # Min trade value
        self.min_trade_value_usd = min_trade_value_usd

        # Phase
        self.phase = phase

        # Max lookback for VP + features
        self.max_lookback = max(max(d * 24 for d in self.vp_days), 50) + self.lookback_window
        logging.info(f"Max Lookback Set To: {self.max_lookback} steps")

    def get_feature_names(self):
        """Returns list of feature names for saliency analysis."""
        names = self.features * self.lookback_window
        names += ["balance_norm", "holdings_norm"]
        for day in self.vp_days:
            p = f"vp_{day}d"
            names += [f"{p}_dist_poc", f"{p}_dist_vah", f"{p}_dist_val"]
            names += [f"{p}_bucket_{i}" for i in range(self.vp_bins)]
        names += self.div_features
        names += [f"recent_action_{i}" for i in range(5)]
        names += [f"position_delta_{i}" for i in range(5)]
        return names

    def _detect_divergences(self, osc_idx, price_idx, window=40, tolerance=8):
        """Detects bullish/bearish divergences between oscillator and price."""
        if self.current_step < window + 20:
            return 0.0, 0.0
        start = max(0, self.current_step - window)
        prices = self.data_matrix[start:self.current_step + 1, price_idx]
        osc = self.data_matrix[start:self.current_step + 1, osc_idx]
        lows_idx = argrelextrema(prices, np.less, order=3)[0]
        highs_idx = argrelextrema(prices, np.greater, order=3)[0]
        bullish = bearish = 0.0
        recent = lambda i: (len(prices) - 1 - i) <= tolerance
        if len(lows_idx) >= 2 and recent(lows_idx[-1]) and prices[lows_idx[-1]] < prices[lows_idx[-2]]:
            near1 = [i for i in argrelextrema(osc, np.less, order=3)[0] if abs(i - lows_idx[-2]) <= 8]
            near2 = [i for i in argrelextrema(osc, np.less, order=3)[0] if abs(i - lows_idx[-1]) <= 8]
            if near1 and near2 and osc[near2[0]] > osc[near1[0]]:
                bullish = 1.0
        if len(highs_idx) >= 2 and recent(highs_idx[-1]) and prices[highs_idx[-1]] > prices[highs_idx[-2]]:
            near1 = [i for i in argrelextrema(osc, np.greater, order=3)[0] if abs(i - highs_idx[-2]) <= 8]
            near2 = [i for i in argrelextrema(osc, np.greater, order=3)[0] if abs(i - highs_idx[-1]) <= 8]
            if near1 and near2 and osc[near2[0]] < osc[near1[0]]:
                bearish = 1.0
        return bullish, bearish

    def _sync_wallet_balance(self):
        """Update net_worth based on current holdings and price."""
        current_price = self.raw_prices[self.current_step]
        self.net_worth = self.balance + (self.shares_held * current_price)

    def _take_action(self, action):
        """Execute trade based on a signed target exposure."""
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])

        # --- NEW: Hard Cooldown Enforcer ---
        # If we traded recently, force the agent to stick to its previous side (Long or Short)
        if self.steps_since_last_trade < self.cooldown_steps and self.trades_in_episode > 0:
            current_sign = np.sign(self.shares_held) if abs(self.shares_held) > 1e-6 else 0
            new_sign = np.sign(action_val) if abs(action_val) > 1e-6 else 0
            # If trying to flip direction during cooldown, block it
            if current_sign != 0 and new_sign != 0 and current_sign != new_sign:
                return False

        # Target position in USD (leverage scaled)
        target_usd = action_val * self.net_worth * self.max_leverage * self.leverage_buffer

        # Current position value
        current_usd = self.shares_held * current_price

        # Required trade size
        trade_usd = target_usd - current_usd

        # Apply minimum trade size filter
        if abs(trade_usd) < self.min_trade_value_usd:
            return False

        # Execute trade
        shares_to_trade = trade_usd / current_price
        cost = abs(shares_to_trade * current_price) * self.transaction_cost_rate

        trade_occurred = False
        if trade_usd > 0:  # Buy
            if self.balance >= shares_to_trade * current_price + cost:
                self.shares_held += shares_to_trade
                self.balance -= shares_to_trade * current_price + cost
                if self.shares_held > 0 and self.entry_price == 0.0:  # NEW: Set entry on first buy
                    self.entry_price = current_price
                trade_occurred = True
                self.trades_in_episode += 1
        elif trade_usd < 0:  # Sell / Short-cover
            if self.shares_held >= -shares_to_trade:
                self.shares_held += shares_to_trade  # shares_to_trade is negative
                self.balance -= shares_to_trade * current_price - cost  # credit for sell
                trade_occurred = True
                self.trades_in_episode += 1

        if trade_occurred:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1

        return trade_occurred

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)

        # CRITICAL FIX: Force modern Generator (integers() works)
        # Gymnasium sometimes gives old RandomState → we override it
        if not hasattr(self.np_random, 'integers'):
            self.np_random = np.random.default_rng(seed)        

        # Reset financial state
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0.0
        self.prev_shares_held = 0.0
        self.entry_price = 0.0  # NEW: Reset entry price
        self.current_position = 0

        # Reset episode trackers
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.trades_in_episode = 0
        self.steps_since_last_trade = 0
        self.has_traded_once = False
        self.prev_action = 0.0  # NEW: Reset prev action
        self.history_net_worth = [self.initial_balance]
        self.portfolio_returns = deque(maxlen=100)
        self.returns = deque(maxlen=100)
        self.recent_actions = deque(maxlen=5)
        self.recent_position_deltas = deque(maxlen=5)
        self.action_history = []

        # Random start position
        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback

        obs = self._next_observation()
        return obs, {}  # Gymnasium tuple

    def _next_observation(self):
        """Build observation vector."""
        # Market features (lookback window)
        start = max(0, self.current_step - self.lookback_window)
        market = self.market_features[start:self.current_step].flatten()

        # Account state
        current_price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * current_price) / self.initial_balance
        account = np.array([balance_norm, holdings_norm], dtype=np.float32)

        # VP features
        vp_parts = []
        for days in self.vp_days:
            d = self.vp_data[days]
            idx = min(self.current_step, len(d['poc']) - 1)
            poc = d['poc'][idx]
            vah = d['vah'][idx]
            val = d['val'][idx]
            heatmap = d['heatmap'][idx].astype(np.float32)
            total = np.sum(heatmap)
            if total > 0:
                heatmap = heatmap / total

            dist_poc = (poc - current_price) / (current_price + 1e-8)
            dist_vah = (vah - current_price) / (current_price + 1e-8)
            dist_val = (val - current_price) / (current_price + 1e-8)

            vp_parts.extend([dist_poc, dist_vah, dist_val])
            vp_parts.extend(heatmap)
        vp_vec = np.array(vp_parts, dtype=np.float32)

        # Update divergences (decay old signals)
        for k in self.div_scores:
            self.div_scores[k] *= 0.95  # Decay
        bull9, bear9 = self._detect_divergences(self.stoch_rsi_norm_idx, self.close_pct_idx, window=9)
        bull14, bear14 = self._detect_divergences(self.stoch_14_idx, self.close_pct_idx, window=14)
        bull_rsi, bear_rsi = self._detect_divergences(self.rsi_norm_idx, self.close_pct_idx)
        if bull9 > 0: self.div_scores['bull_div_stoch9'] = 1.0
        if bear9 > 0: self.div_scores['bear_div_stoch9'] = 1.0
        if bull14 > 0: self.div_scores['bull_div_stoch14'] = 1.0
        if bear14 > 0: self.div_scores['bear_div_stoch14'] = 1.0
        if bull_rsi > 0: self.div_scores['bull_div_rsi'] = 1.0
        if bear_rsi > 0: self.div_scores['bear_div_rsi'] = 1.0
        div_vec = np.array([self.div_scores[k] for k in self.div_features], dtype=np.float32)

        # Recurrent: recent actions + position deltas
        act_pad = np.array(list(self.recent_actions) + [0.0] * (5 - len(self.recent_actions)), dtype=np.float32)
        delta_pad = np.array(list(self.recent_position_deltas) + [0.0] * (5 - len(self.recent_position_deltas)), dtype=np.float32)

        # Full obs
        full_obs = np.concatenate([market, account, vp_vec, div_vec, act_pad, delta_pad])
        return full_obs.astype(np.float32)

    def step(self, action):
        """Execute one time step."""
        self.current_step += 1
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])

        # NEW: Sync net worth before action
        self._sync_wallet_balance()
        prev_net_worth = self.net_worth

        # NEW: Real portfolio return (no look-ahead)
        portfolio_return = (self.net_worth - self.prev_net_worth) / (self.prev_net_worth + 1e-8)

        # Execute action
        self.prev_shares_held = self.shares_held
        trade_occurred = self._take_action(action)

        # NEW: Real transaction cost penalty
        trade_cost = self.transaction_cost_rate if trade_occurred else 0.0
        # We magnify this in the reward to force high conviction
        reward_trade_cost = trade_cost * self.reward_fee_multiplier

        # NEW: Inertia penalty (discourage twitching)
        inertia_penalty = self.action_penalty * abs(action_val - self.prev_action)

        # NEW: Holding bonus (reward stability)
        # Give extra bonus if we are holding a PROFITABLE position
        holding_bonus = 0.0
        if self.shares_held != 0:
            current_val = self.shares_held * current_price
            entry_val = self.shares_held * self.entry_price
            if (self.shares_held > 0 and current_val > entry_val) or \
               (self.shares_held < 0 and current_val < entry_val):
                holding_bonus = 0.0005  # Drip reward for riding a trend

        # NEW: Realized PnL bonus for profitable closes
        if trade_occurred and self.shares_held < self.prev_shares_held and self.entry_price > 0:
            realized_pnl = (current_price - self.entry_price) * (self.prev_shares_held - self.shares_held)
            if realized_pnl > 0:
                reward = (realized_pnl / self.initial_balance) * 0.5  # Scaled bonus
            else:
                reward = (realized_pnl / self.initial_balance) * 0.2  # Milder loss
        else:
            # Base reward: portfolio return minus costs + bonuses
            reward = portfolio_return - reward_trade_cost - inertia_penalty + holding_bonus

        # Hard overtrading cap
        if self.trades_in_episode > 20:  # Increased from 10
            reward -= 0.5

        if trade_occurred:
            self.has_traded_once = True

        # Update trackers
        self.prev_action = action_val  # NEW: For next inertia
        self.recent_actions.append(action_val)
        self.action_history.append(action_val)
        delta = self.shares_held - self.prev_shares_held
        self.recent_position_deltas.append(delta)
        self.portfolio_returns.append(portfolio_return)
        self.returns.append(reward)
        self.history_net_worth.append(self.net_worth)
        self.prev_net_worth = self.net_worth
        self.current_position = np.sign(self.shares_held) if abs(self.shares_held) > 1e-6 else 0.0
        self.last_price = current_price

        # Termination
        terminated = False
        truncated = self.current_step >= len(self.df) - 1
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            reward -= 5.0  # Episode penalty

        obs = self._next_observation()

        # Info dict
        primary_day = self.vp_days[0]
        idx_safe = min(self.current_step, len(self.vp_data[primary_day]['heatmap']) - 1)
        heatmap = self.vp_data[primary_day]['heatmap'][idx_safe]

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "reward": reward,
            "price": current_price,
            "current_price": current_price,
            "ema50": self.data_matrix[self.current_step, self.ema_50_idx],
            "timestamp": str(self.raw_df.index[self.current_step]),
            "vp_heatmap": heatmap,
            "trades_per_episode": self.trades_in_episode,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human', title_suffix=""):
        if len(self.history_net_worth) < 2:
            return None

        steps = np.arange(len(self.history_net_worth))
        prices = self.raw_prices[:len(self.history_net_worth)]
        actions = self.action_history
        net_worths = self.history_net_worth

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Price
        ax1.plot(steps, prices, label='Price', color='black', linewidth=1.2)
        ax1.set_title(f"Trade Analysis{title_suffix}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Actions
        colors = ['green' if a > 0 else 'red' for a in actions]
        ax2.bar(steps, actions, color=colors, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Action")
        ax2.grid(True, alpha=0.3)

        # Net Worth
        ax3.plot(steps, net_worths, label='Net Worth', color='blue', linewidth=1.2)
        ax3.set_ylabel("Net Worth")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # FIX FOR MULTIPROCESSING: Convert Figure to RGB Array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return data  # Return the Numpy Array (Safe for SubprocVecEnv)

    # Phase setters (for curriculum)
    def set_phase(self, new_phase):
        self.phase = new_phase

    def set_min_trade_value(self, value):
        self.min_trade_value_usd = float(value)

    def set_thresholds(self, buy, sell):
        self.buy_threshold = buy
        self.sell_threshold = sell

# Register the environment with Gym
gym.register(
    id='EnhancedTradingEnv-v0',
    entry_point='enhanced_trading_env:EnhancedTradingEnv',
    # Note: df must be provided when creating the env
)