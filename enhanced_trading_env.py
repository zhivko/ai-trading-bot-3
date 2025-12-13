import glob
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
from scipy.signal import argrelextrema
import os
import matplotlib.pyplot as plt

# Delegating heavy lifting to volume_profile.py
from volume_profile import get_rolling_vp
from phase_manager import PhaseManager
import logging



def get_thread_logger():
    """Create a logger that writes to a separate file per thread/environment instance."""
    import threading
    import logging

    # Get thread identifier (process ID + thread ID for uniqueness)
    thread_id = f"{os.getpid()}_{threading.current_thread().ident}"

    # Create logger name based on thread
    logger_name = f"EnhancedTradingEnv_{thread_id}"

    # Get or create logger
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler for this thread's log file
        log_file = os.path.join(log_dir, f"env_{thread_id}.log")

        # Use a lock to prevent race conditions when deleting/creating files
        import threading
        lock = threading.Lock()
        with lock:
            # Delete existing log file if it exists to start fresh
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except OSError:
                    pass  # Ignore if deletion fails

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

    return logger


class EnhancedTradingEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'render_fps': 4,
    }

    def __init__(self, df, initial_balance=10000, lookback_window=50, vp_days=None, vp_bins=40,
                         buy_threshold=0.5, sell_threshold=-0.5, precalculated_vp=None, trading_fee_multiplier=0.00075, phase=1, total_phases=10, min_trade_value_usd=10.0,
                         pair='BTCUSDT', timeframe='1h', split_date=None, phase_manager=None):
        try:
            # Create instance-specific logger
            self._logger = get_thread_logger()
            self._logger.info("EnhancedTradingEnv instance created")
            self._logger.info(f"--- Initializing EnhancedTradingEnv (Target Bins: {vp_bins}) ---")
            super(EnhancedTradingEnv, self).__init__()
            self._logger.info("Super init completed")
        except Exception as e:
            print(f"Error during super init: {e}")
            raise

        # Update metadata with instance-specific info
        self.metadata.update({
            'pair': pair,
            'timeframe': timeframe,
            'data_range': f"{df.index[0]} to {df.index[-1]}" if not df.empty else None,
            'split_date': split_date,
        })
        self._logger.info("Metadata updated")

        # === OVERTRADING FIXES ===
        self.transaction_cost_rate = 0.0015      # 0.15% per trade (Binance spot taker fee ≈ 0.1% + slippage)
        self.reward_fee_multiplier = 1.5         # Amplify fee impact in reward
        self.action_penalty = 0.0005             # Moderate — allows switching but punishes noise
        self.holding_penalty = 0.0        # REMOVED: Was 0.0005, now 0.0 to allow holding profitable positions
        self.trade_penalty = 0.05                 # Higher fixed cost per trade
        self.last_trade_cost = 0
        self.reward_trade_cost = 0.0
        self.steps_in_trade = 0
        self.last_trade_pnl = 0.0
        
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
            self._logger.info(f"--- Initializing EnhancedTradingEnv (Target Bins: {self.vp_bins}) ---")
            
            for days in self.vp_days:
                # All caching/hashing logic is now in volume_profile.py
                self.vp_data[days] = get_rolling_vp(self.raw_df, days, bins=self.vp_bins)
        
        self._logger.info("Standard features...")
        # Standard Features
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)

        # --- FIX: Rolling Max Normalization ---
        # Instead of Z-score or simple Log, we scale volume relative to the last 100 steps.
        # This ensures the value is almost always between 0.0 and 1.0

        self._logger.info("Calculating rolling max volume for normalization...")
        vol_rolling_max = self.df['volume'].rolling(window=100, min_periods=1).max()

        # 2. Divide current volume by rolling max (Safe division)
        self.df['volume_norm'] = self.df['volume'] / (vol_rolling_max + 1e-8)

        # 3. Fill NaNs just in case
        self.df['volume_norm'] = self.df['volume_norm'].fillna(0)

        # Indicators
        self._logger.info("Indicators...")
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

        # NEW: Fast vs Slow Trend Alignment
        ema_20 = self.df['close'].ewm(span=20, adjust=False).mean()
        # Normalized difference between Fast and Slow EMA
        # When this turns negative, it's an early warning that the trend is breaking
        self.df['trend_fast_diff'] = (ema_20 - self.df['ema_50']) / (self.df['ema_50'] + 1e-8)

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

        self.features = ['close_pct', 'volume_norm', 'rsi_norm', 'stoch_rsi_norm', 'macd_norm', 'macd_sig_norm', 'trend_ema_norm', 'atr_norm', 'regime', 'trend_fast_diff']
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
        self._logger.info("Column indices...")
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
        self._logger.info("Market features...")
        feature_indices = [self.df.columns.get_loc(f) for f in self.features]
        self.market_features = self.data_matrix[:, feature_indices]
        self.raw_prices = self.data_matrix[:, self.df.columns.get_loc('close')]

        # --- FIX: Initialize feature names explicitly for Main.py ---
        self.feature_names = self.get_feature_names()

        # --- 3. SPACE DEFINITION (Updated for new features) ---
        self._logger.info("Space definition...")
        market_obs_size = self.lookback_window * len(self.features)
        account_obs_size = 6
        vp_obs_size = len(self.vp_days) * (3 + self.vp_bins)
        recurrent_obs_size = 12  # 5 recent actions + 5 position deltas + 2 extra features
        total_obs_size = (
            market_obs_size
            + account_obs_size
            + vp_obs_size
            + len(self.div_features)
            + recurrent_obs_size
        )

        # Action now encodes signed target exposure in [-1, 1]
        # -1 = max short, 0 = flat, 1 = max long (before leverage scaling)
        self._logger.info("Defining action and observation spaces...")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32)

        self.entry_price = 0.0  # + NEW: Initialize to fix AttributeError in step

        # === FIX: Soften churn/cooldown to not over-punish trading ===
        self.target_hold_duration = 24       # Target 24h hold
        self.cooldown_steps = 0              # REMOVED: Hard constraints confuse the agent. Let fees dictate holding.

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

        # Min trade value - REDUCED: Allow smaller position adjustments
        self.min_trade_value_usd = 5.0 if min_trade_value_usd == 10.0 else min_trade_value_usd

        # Phase
        self.phase = phase
        self.total_phases = max(1, total_phases)  # Safety: ensure at least 1
        self.phase_manager = phase_manager if phase_manager else PhaseManager(total_phases=total_phases)

        # === DYNAMIC THRESHOLD LOGIC ===
        phase_params = self.phase_manager.get_phase_params(self.phase)
        self.buy_threshold = phase_params['buy_threshold']
        self.sell_threshold = phase_params['sell_threshold']
        self._logger.info(f"Phase {self.phase}/{self.total_phases} :: thresholds: {self.sell_threshold:.4f} / {self.buy_threshold:.4f}")

        # Log it so you can verify it's working
        self._logger.info(f"Phase {self.phase}/{self.total_phases} :: thresholds: {self.sell_threshold:.4f} / {self.buy_threshold:.4f}")

        # Max lookback for VP + features
        self.max_lookback = max(max(d * 24 for d in self.vp_days), 50) + self.lookback_window
        self._logger.info(f"Max Lookback Set To: {self.max_lookback} steps")
        self._logger.info("EnhancedTradingEnv initialization completed successfully")

    def get_feature_names(self):
        """Returns list of feature names for saliency analysis."""
        names = self.features * self.lookback_window
        names += ["balance_norm", "holdings_norm", "pos_flag", "unrealized_pnl", "time_held", "last_pnl"]
        for day in self.vp_days:
            p = f"vp_{day}d"
            names += [f"{p}_dist_poc", f"{p}_dist_vah", f"{p}_dist_val"]
            names += [f"{p}_bucket_{i}" for i in range(self.vp_bins)]
        names += self.div_features
        names += [f"recent_action_{i}" for i in range(5)]
        names += [f"position_delta_{i}" for i in range(5)]
        names += ["time_held_feature", "current_pnl_pct"]
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

        # Calculate target equity based on action
        target_equity = action_val * self.net_worth * self.max_leverage

        # Calculate deviation from current holding
        current_equity = self.shares_held * current_price
        deviation = target_equity - current_equity

        # === FIX: Close Dust Positions ===
        # If target is ~0 (Close Signal), we ignore min_trade_value to ensure clean exit.
        # Otherwise, we enforce min_trade_value to prevent noise trading.
        is_close_signal = np.isclose(target_equity, 0, atol=1.0) # Treat <$1 target as 0

        if abs(deviation) > self.min_trade_value_usd or (is_close_signal and abs(self.shares_held) > 0):
             # If deviation is negative, we Sell. If positive, we Buy.
             # PASS force_close=True if it is a close signal to bypass min_trade_value inside execute
             return self._execute_trade(deviation, force_close=is_close_signal)

        return False

    def _execute_trade(self, deviation, force_close=False):
        """Execute a trade based on the deviation amount."""
        current_price = self.raw_prices[self.current_step]

        if deviation > 0:  # Buy
            trade_usd = deviation

            # Clamp to available balance
            if trade_usd > self.balance:
                trade_usd = self.balance

            # Double check min size after clamping
            if trade_usd < self.min_trade_value_usd:
                return False            

            shares_to_buy = trade_usd / current_price
            current_shares = self.shares_held
            new_shares = current_shares + shares_to_buy

            # Entry price update
            if current_shares > 0:
                old_cost = current_shares * self.entry_price
                new_buy_cost = shares_to_buy * current_price
                self.entry_price = (old_cost + new_buy_cost) / new_shares
            elif current_shares < 0 and new_shares > 0:
                self.entry_price = current_price
            elif current_shares == 0:
                self.entry_price = current_price

            # Execute
            self.balance -= trade_usd
            self.shares_held += shares_to_buy
            self.trades_in_episode += 1
            self.steps_since_last_trade = 0
            return True

        elif deviation < 0:  # Sell
            trade_usd = -deviation

            # Calculate max allowed sell
            current_shares = self.shares_held
            value_of_longs = (current_shares * current_price) if current_shares > 0 else 0.0
            short_exposure = abs(current_shares * current_price) if current_shares < 0 else 0.0
            available_margin = self.net_worth - short_exposure
            if available_margin < 0:
                available_margin = 0.0
            max_sell = value_of_longs + available_margin

            # Clamp
            if trade_usd > max_sell:
                trade_usd = max_sell

            # Double check min size after clamping
            if trade_usd < self.min_trade_value_usd:
                return False

            shares_to_sell = trade_usd / current_price
            new_shares = current_shares - shares_to_sell

            # Entry price update
            if current_shares < 0:
                old_cost = current_shares * self.entry_price
                new_sell_cost = -(shares_to_sell * current_price)
                self.entry_price = (old_cost + new_sell_cost) / new_shares
            elif current_shares > 0 and new_shares < 0:
                self.entry_price = current_price

            # Execute
            self.balance += trade_usd
            self.shares_held -= shares_to_sell
            self.trades_in_episode += 1
            self.steps_since_last_trade = 0
            return True

        return False

    # Critical: add .seed() method that SB3 expects
    def seed(self, seed=None):
        return seed

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=self.seed())

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
        self.reward_trade_cost = 0.0
        self.steps_in_trade = 0
        self.last_trade_pnl = 0.0
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

        # Ensure we don't start beyond the data
        self.current_step = min(self.current_step, len(self.df) - 1)

        obs = self._next_observation()
        return obs, {}  # Gymnasium tuple

    def _next_observation(self):
        """Build observation vector."""
        # Market features (lookback window)
        # Get the data for the current window
        # Shape: (window_size, n_features)

        # FIX: .iloc requires numeric indexers. We select columns by name first, then slice rows by position.
        start_idx = self.current_step - self.lookback_window + 1
        end_idx = self.current_step + 1
        if start_idx < 0 or end_idx > len(self.df):
            # Handle edge case where we don't have enough history
            # Pad with zeros or repeat first/last values
            window_size = self.lookback_window
            if start_idx < 0:
                # Pad beginning with first available data
                available_start = max(0, start_idx)
                pad_size = abs(start_idx)
                available_data = self.df[self.features].iloc[available_start:end_idx].values
                if len(available_data) > 0:
                    pad_data = np.tile(available_data[0], (pad_size, 1))
                    raw_obs = np.vstack([pad_data, available_data])
                else:
                    raw_obs = np.zeros((window_size, len(self.features)))
            else:
                raw_obs = self.df[self.features].iloc[start_idx:end_idx].values
        else:
            raw_obs = self.df[self.features].iloc[start_idx:end_idx].values
        
        # IMPLEMENTATION: Windowed Z-Score Normalization
        # This ensures all inputs (Price, RSI, Volume) are on the same scale (-2 to +2 range)
        # centered around 0 for the specific window the agent is looking at.
        
        # Calculate mean and std for each column in this window
        means = np.mean(raw_obs, axis=0)
        stds = np.std(raw_obs, axis=0)
        
        # Add a small epsilon to prevent division by zero (if a column is constant)
        stds[stds == 0] = 1e-8
        
        # Normalize: (Value - Mean) / StdDev
        normalized_obs = (raw_obs - means) / stds
        
        market = normalized_obs.flatten()

        # Account state
        current_price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * current_price) / self.initial_balance
        pos_flag = 1.0 if self.shares_held > 0 else -1.0 if self.shares_held < 0 else 0.0
        unrealized_pnl = ((current_price - self.entry_price) * self.shares_held) / self.initial_balance if self.shares_held != 0 else 0.0
        time_held = self.steps_in_trade / 100.0
        last_pnl = self.last_trade_pnl / self.initial_balance
        account = np.array([balance_norm, holdings_norm, pos_flag, unrealized_pnl, time_held, last_pnl], dtype=np.float32)

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

        time_held_feature = self.steps_in_trade / 100.0
        current_pnl_pct = ((current_price - self.entry_price) * self.shares_held) / (abs(self.entry_price * self.shares_held) + 1e-8) if self.shares_held != 0 else 0.0

        # Full obs
        full_obs = np.concatenate([market, account, vp_vec, div_vec, act_pad, delta_pad, np.array([time_held_feature, current_pnl_pct], dtype=np.float32)])
        return full_obs.astype(np.float32)

    def step(self, action):
        """Execute one time step."""
        # Check if episode should end before incrementing
        if self.current_step >= len(self.df) - 1:
            obs = self._next_observation()
            return obs, 0.0, False, True, {}

        self.current_step += 1
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])

        # NEW: Sync net worth before action
        self._sync_wallet_balance()
        prev_net_worth = self.net_worth

        # NEW: Real portfolio return (no look-ahead)
        portfolio_return = (self.net_worth - self.prev_net_worth) / (self.prev_net_worth + 1e-8)

        # === THRESHOLD CHECK ===
        # Always attempt to adjust position to the target.
        # The _take_action method already checks 'min_trade_value_usd' to prevent dust trades.
        trade_occurred = self._take_action(action)

        # === FIX STARTS HERE ===
        # 1. Calculate the actual size of the trade in dollars
        shares_diff = abs(self.shares_held - self.prev_shares_held)
        trade_value = shares_diff * current_price

        # 2. Calculate the actual dollar cost (Fee)
        actual_cost = trade_value * self.transaction_cost_rate

        # 3. Convert to Percentage of Net Worth
        # (Because your main reward is percentage-based, the penalty must match)
        cost_pct = 0.0
        if self.net_worth > 0:
            cost_pct = actual_cost / self.net_worth

        # 4. Apply the Multiplier (The "Pain" Factor)
        # We multiply by 100 to match the 'reward' scale (where 1% = 1.0)
        reward_trade_cost = (cost_pct * 100.0) * self.reward_fee_multiplier
        # === FIX ENDS HERE ===

        # --- UPDATE AGENT STATE ---
        if abs(self.shares_held) > 0:
            self.steps_in_trade += 1
        else:
            self.steps_in_trade = 0

        # --- REWARD CALCULATION ---

        # 1. Base Reward: Net Worth Change (Captures Unrealized PnL naturally)
        # If price goes up while holding, this is positive. If price goes down, this is negative.
        reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 100.0
        reward_base = reward  # net worth change component

        # 2. Subtract costs (Fee is now reduced to 2x multiplier in init)
        reward -= reward_trade_cost  # Now this variable is definitely defined

        # 3. Action change penalty (discourage twitching)
        action_change_penalty = 0.0
        if self.prev_action is not None:
            # Calculate raw change in action (0.0 to 2.0 range)
            action_delta = abs(action_val - self.prev_action)
            # REDUCED: 0.05 -> 0.02 to allow more flexibility
            action_change_penalty = -(action_delta * 0.02)  # negative penalty
        reward += action_change_penalty  # adding negative subtracts

        # 4. Inertia penalty (discourage twitching without position change)
        reward_inertia = 0.0
        if self.shares_held == self.prev_shares_held and abs(action_val) > 0.5:
             reward_inertia = -0.05  # negative penalty
        reward += reward_inertia

        # --- FIX: TREND ALIGNMENT PENALTY ---
        # Punish holding Long in Bear market or Short in Bull market
        # This fixes "Saturation" where agent ignores crashes
        ema_50 = self.data_matrix[self.current_step, self.ema_50_idx]
        # 1 = Bull, -1 = Bear
        trend_direction = 1.0 if current_price > ema_50 else -1.0
        # If position direction opposes trend direction (and not cash)
        trend_penalty = 0.0
        if self.shares_held != 0 and np.sign(self.shares_held) != trend_direction:
            trend_penalty = 0.05
            reward -= trend_penalty  # Constant penalty per step for fighting the trend

        # NEW: Asymmetric holding penalty — punish staying long when trend is weakening/bearish
        # Encourages early exit/reversal in downtrends without punishing bull holds
        if self.current_position > 0.1 and self.df.iloc[self.current_step]['trend_ema_norm'] < 0.2:
            weakening = max(0, 0.2 - self.df.iloc[self.current_step]['trend_ema_norm'])
            reward -= 0.005 * weakening ** 2   # Stronger quadratic penalty as trend weakens

        # Calculate "Rent" (Funding Fee) to discourage camping on a position
        current_holding_cost = 0.0
        if self.shares_held != 0:
            current_holding_cost = self.holding_penalty

        reward -= current_holding_cost  # Apply the rent
        
        # Add small deadzone penalty around action=0 in reward (discourage tiny noisy actions but allow clean holds)
        reward -= 0.0002 * (abs(action_val) < 0.1) * (1 - abs(action_val)/0.1)  # Small penalty for lingering near zero

        # 3. "Closer's Bonus" (Realized PnL Stimulus)
        closer_bonus = 0.0
        if self.prev_shares_held != 0 and self.shares_held == 0:
             realized_pnl_val = (current_price - self.entry_price) * self.prev_shares_held
             trade_return_pct = realized_pnl_val / abs(self.entry_price * self.prev_shares_held + 1e-8)

             # Store for next observation
             self.last_trade_pnl = trade_return_pct

             if realized_pnl_val > 0:
                 # BONUS: Reward realizing a win 2x more than just holding it
                 closer_bonus = (trade_return_pct * 100.0) * 2.0
             else:
                 # Standard penalty for realizing a loss
                 closer_bonus = (trade_return_pct * 100.0) * 1.0
             reward += closer_bonus
             
             # Reset entry price
             self.entry_price = 0.0

        # 4. FIX: Death Spiral Bug
        # Only penalize overtrading IF a trade actually occurred this step
        # INCREASED: 20 -> 50 to allow more active trading strategy
        overtrading_penalty = 0.0
        if trade_occurred and self.trades_in_episode > 50:
            overtrading_penalty = 0.5
            reward -= overtrading_penalty

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
        episode_penalty = 0.0
        truncated = bool(self.current_step >= len(self.df) - 1)
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            episode_penalty = 5.0
            reward -= episode_penalty  # Episode penalty

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
            "trade_executed": trade_occurred,
            # Reward components
            "reward_base": reward_base,
            "reward_fee": reward_trade_cost,
            "reward_action_change": action_change_penalty,
            "reward_trend": trend_penalty,
            "reward_holding": current_holding_cost,
            "reward_inertia": reward_inertia,
            "reward_closer": closer_bonus,
            "reward_overtrade": overtrading_penalty,
            "reward_episode": episode_penalty,
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
        try:
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data  # Return the Numpy Array (Safe for SubprocVecEnv)
        except Exception as e:
            self._logger.warning(f"Render failed: {e}")
            plt.close(fig)
            return None

    # Phase setters (for curriculum)
    def set_phase(self, new_phase):
        self.phase = new_phase
        # Update thresholds when phase changes
        phase_params = self.phase_manager.get_phase_params(self.phase)
        self.buy_threshold = phase_params['buy_threshold']
        self.sell_threshold = phase_params['sell_threshold']
        self._logger.info(f"Phase updated to {self.phase}: buy_threshold={self.buy_threshold:.3f}, sell_threshold={self.sell_threshold:.3f}")

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