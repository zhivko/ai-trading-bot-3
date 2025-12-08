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
        super(EnhancedTradingEnv, self).__init__()

        # Update metadata with instance-specific info
        self.metadata.update({
            'pair': pair,
            'timeframe': timeframe,
            'data_range': f"{df.index[0]} to {df.index[-1]}" if not df.empty else None,
            'split_date': split_date,
        })

    def __init__(self, df, initial_balance=10000, lookback_window=50, vp_days=None, vp_bins=40,
                      buy_threshold=0.5, sell_threshold=-0.5, precalculated_vp=None, trading_fee_multiplier=0.00075, phase=1, min_trade_value_usd=10.0,
                      pair='BTCUSDT', timeframe='1h', split_date=None):
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
        self.action_penalty = 0.0005             # tiny L1 penalty to discourage twitching
        self.last_trade_cost = 0
        
        # --- CONFIGURATION ---
        self.initial_balance = initial_balance
        self.last_price = 0
        self.current_position = 0
        # === LOOK-AHEAD REWARD SHAPING (the nuclear weapon against overtrading) ===
        # How many bars into the future we peek: 1h, 4h, 12h, 24h on 1h data → indices 1,4,12,24
        self.lookahead_horizons = [1, 4, 12, 24]          # adjust if you use 4h or daily data
        self.lookahead_weights  = [0.4, 0.3, 0.2, 0.1]     # more weight on near-term
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
        self.reward_scaling = 1.0

        # === FIX: Soften churn/cooldown to not over-punish trading ===
        self.target_hold_duration = 0
        self.cooldown_steps = 1              # Allow frequent trading for learning

        # REWARD PENALTIES
        # We switch to a PURE PnL model, but we add a fixed "Cost of Living"
        # for trading to simulate spread/slippage anxiety.
        self.trade_fee_penalty = 0.5

        self.prev_actions = deque(maxlen=3)
        self.action_history = []  # For rendering the full action chart
        self.returns = []
        self.portfolio_returns = deque(maxlen=100)
        self.recent_actions = deque(maxlen=5)
        self.recent_position_deltas = deque(maxlen=5)

        self.max_lookback = max(max([d * 24 for d in self.vp_days]), 30) + self.lookback_window

        self.phase = phase
        self.min_trade_value_usd = float(min_trade_value_usd)

        # --- Leverage & shorting parameters ---
        self.max_leverage = 5.0          # allow 1x .. 5x long/short
        self.leverage_buffer = 0.95      # small safety margin

        self.min_action_threshold = 0.0    # No deadband

        # === FIX: Anti-zero-reward params ===
        self.hold_bonus = 0.0  # Remove: No free positive for flat holds
        self.exploration_bonus = 0.005  # Keep but lower; apply only on first trade
        self.has_traded_once = False  # Track to fade bonus

        self.prev_sign = 0  # Initialize for switch penalty
        self.hold_steps = 0  # Track hold duration for churn penalty
        self.last_trade_step = 0  # For churn calculation
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

    def _sync_wallet_balance(self):
        # Placeholder for syncing wallet balance via API
        # In a real implementation, this would call an API to get current balance
        # and update self.balance accordingly, handling dust positions or partial fills.
        logging.info("Syncing wallet balance via API")
        # Example: self.balance = api_client.get_balance('USDT')
        # For now, it's a no-op as this is a simulated environment.

    def get_feature_names(self):
        """
        Returns a list of labels that EXACTLY matches the observation vector construction.
        """
        # 1. Market Features (from self.features in __init__)
        # This guarantees the order matches self.market_features
        names = self.features.copy()
        
        # 2. Account Features (Added manually in _next_observation)
        names += ["balance_norm", "holdings_norm"]
        
        # 3. Volume Profile Features (Loop matches _next_observation logic)
        for day in self.vp_days:
            # Distances (poc, vah, val) added before heatmap
            prefix = f"vp_{day}d"
            names += [f"{prefix}_dist_poc", f"{prefix}_dist_vah", f"{prefix}_dist_val"]
            
            # Heatmap Buckets
            names += [f"{prefix}_bucket_{i}" for i in range(self.vp_bins)]
            
        # 4. Divergence Features
        names += self.div_features

        # 5. Recurrent Features
        names += [f"recent_action_{i}" for i in range(5)]
        names += [f"position_delta_{i}" for i in range(5)]

        return names

def _take_action(self, action):
    """
    Execute trade based on a signed target exposure:
    action[0] in [-1, 1] → target_leverage in [-max_leverage, +max_leverage].

    Negative target_leverage = net short, positive = net long.
    self.shares_held is allowed to be negative to represent shorts.
    """
    self.last_trade_cost = 0.0
    self.action_was_blocked = False

    raw = float(action[0])
    raw = np.clip(raw, -1.0, 1.0)

    # Map raw action to leveraged target exposure (e.g. -5 .. +5)
    target_leverage = raw * self.max_leverage

    current_price = self.raw_prices[self.current_step]
    position_value = self.shares_held * current_price  # can be negative

    equity = self.balance + position_value
    if equity <= 0:
        logging.warning("Equity <= 0, skipping trade.")
        return False

    current_leverage = position_value / equity  # can be negative

    # Respect leverage limit with a small safety buffer
    max_abs = self.max_leverage * self.leverage_buffer
    target_leverage = float(np.clip(target_leverage, -max_abs, max_abs))

    delta_lev = target_leverage - current_leverage
    if abs(delta_lev) < 1e-6:
        # No meaningful change requested
        self.net_worth = equity
        self.history_net_worth.append(self.net_worth)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        return False

    # Desired change in position value (can be positive or negative)
    trade_value = delta_lev * equity

    if abs(trade_value) < self.min_trade_value_usd:
        logging.debug(
            f"Trade blocked: |trade_value|={abs(trade_value):.2f} "
            f"< min_trade_value_usd={self.min_trade_value_usd}"
        )
        self.action_was_blocked = True
        self.net_worth = equity
        self.history_net_worth.append(self.net_worth)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        return False

    shares_to_trade = trade_value / current_price
    trade_occurred = False

    # Slippage via VP (same structure you already use)
    primary_day = self.vp_days[0]
    data = self.vp_data[primary_day]
    heatmap = data["heatmap"][self.current_step]
    vp_max = np.max(heatmap)
    slippage = 0.001 * (1 - vp_max)

    if trade_value > 0:
        # Buy more / reduce short
        effective_price = current_price * (1 + slippage)
        cost = shares_to_trade * effective_price
        fee = cost * self.trading_fee_multiplier

        if self.balance >= cost + fee:
            self.balance -= cost + fee
            self.shares_held += shares_to_trade
            self.last_trade_cost = fee
            trade_occurred = True
            self.trades_in_episode += 1
            logging.debug(
                f"BUY / REDUCE SHORT: cost={cost:.2f}, shares={shares_to_trade:.6f}, "
                f"new_shares={self.shares_held:.6f}, target_lev={target_leverage:.2f}"
            )
        else:
            logging.debug(
                f"Buy blocked: need {cost + fee:.2f}, balance={self.balance:.2f}"
            )
            self.action_was_blocked = True
    else:
        # Sell / increase short
        effective_price = current_price * (1 - slippage)
        proceeds = abs(shares_to_trade) * effective_price
        fee = proceeds * self.trading_fee_multiplier

        # For shorts we don't need balance upfront; proceeds increase balance
        self.balance += proceeds - fee
        self.shares_held += shares_to_trade  # shares_to_trade < 0 if increasing short
        self.last_trade_cost = fee
        trade_occurred = True
        self.trades_in_episode += 1
        logging.debug(
            f"SELL / INCREASE SHORT: proceeds={proceeds:.2f}, shares={shares_to_trade:.6f}, "
            f"new_shares={self.shares_held:.6f}, target_lev={target_leverage:.2f}"
        )

    # Recompute after trade and enforce leverage cap (hard stop)
    position_value = self.shares_held * current_price
    equity = self.balance + position_value

    if equity <= 0:
        # Force liquidation to avoid negative equity
        logging.warning("Equity <= 0 after trade, forcing flat.")
        self.balance = max(0.0, self.balance + position_value)
        self.shares_held = 0.0
        position_value = 0.0
        equity = self.balance

    # Hard clamp if leverage drifted over limit
    if equity > 0:
        actual_leverage = abs(position_value) / equity
        if actual_leverage > self.max_leverage * 1.05:
            logging.warning(
                f"Leverage {actual_leverage:.2f} > max_leverage, forcing de‑leveraging."
            )
            # Liquidate to flat
            self.balance = equity
            self.shares_held = 0.0
            position_value = 0.0

    self.net_worth = self.balance + position_value
    self.history_net_worth.append(self.net_worth)
    if self.net_worth > self.max_net_worth:
        self.max_net_worth = self.net_worth

    return trade_occurred

    def compute_reward(self, action, trade_occurred=False):
        """
        Revised Reward: Pure PnL with explicit Fee Punishment.
        """
        # 1. Log Returns (The Truth)
        # We use Log returns because they are symmetric.
        # If NetWorth is 0 (bankruptcy), handle gracefully.
        current_val = max(self.net_worth, 1e-6)
        prev_val = max(self.prev_net_worth, 1e-6)

        # Base Reward = % Change in Portfolio
        # e.g. +1% gain = +1.0 reward
        step_reward = np.log(current_val / prev_val) * 100

        # === FIX 1: Transaction cost + action penalty ===
        step_reward -= self.last_trade_cost
        step_reward -= self.action_penalty * abs(action)

        # 2. Invalid Action Penalty (Teach agent to stop spamming blocked trades)
        if getattr(self, 'action_was_blocked', False):
            step_reward -= 0.5

        # 2. The Fee (Real Cost)
        # If a trade occurred, we subtract a fixed "mental friction" cost
        # on TOP of the actual financial fee loss (which is already in net_worth).
        if trade_occurred:
            step_reward -= self.trade_fee_penalty # e.g. -0.1

        # 3. Holding Bonus (Positive Reinforcement)
        # If we have a position and hold it, give a tiny drip.
        # This helps the agent "wait out" the cooldown without feeling zero reward.
        if abs(self.shares_held) > 0 and not trade_occurred:
            step_reward += 0.005

        # NEW: Dynamic Churn Penalty (on close long or short actions)
        if (action < -0.5 and self.shares_held > 0) or (action > 0.5 and self.shares_held < 0):  # Closing a long or short position
            hold_duration = self.current_step - self.last_trade_step
            if hold_duration > 0:
                if hold_duration < 12:                     # Softer: min hold 12 steps (less punitive)
                    churn_penalty = -0.1 * (1 - hold_duration / 12.0)   # Max -0.1 (was -0.5)
                else:
                    churn_penalty = 0.0
                step_reward += churn_penalty
                logging.debug(f"Churn penalty applied: {churn_penalty:.3f} (hold: {hold_duration})")
            self.hold_steps = 0
            self.last_trade_step = self.current_step
        else:
            self.hold_steps += 1

        # Action smoothing penalty
        prev_action = self.prev_actions[-1] if self.prev_actions else 0
        action_delta = abs(action - prev_action)
        smoothing_penalty = -0.05 * action_delta
        step_reward += smoothing_penalty

        # Cumulative penalty for trades in episode
        step_reward -= 0.05 * self.trades_in_episode

        # Exploration: Only on first trade (not upfront)
        if trade_occurred and not self.has_traded_once:
            exploration = self.exploration_bonus
            self.has_traded_once = True
        else:
            exploration = 0.0
        step_reward += exploration  # No baseline

        # Penalize holding losses aggressively
        if self.shares_held > 0:
            current_price = self.raw_prices[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) * self.shares_held
            if unrealized_pnl < 0:
                step_reward -= unrealized_pnl * 0.1

        # Incorporate risk-adjusted metrics: Sharpe ratio penalty
        if len(self.portfolio_returns) > 10:
            volatility = np.std(list(self.portfolio_returns))
            step_reward -= volatility * 0.1  # lambda = 0.1

        return step_reward

    def set_phase(self, new_phase):
        self.phase = new_phase

    def set_min_trade_value(self, value):
        self.min_trade_value_usd = float(value)

    def set_thresholds(self, buy, sell):
        self.buy_threshold = buy
        self.sell_threshold = sell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self._sync_wallet_balance()  # Sync balance on reset
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0
        self.prev_shares_held = 0
        self.last_price = 0
        self.current_position = 0
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.prev_action = 0
        self.prev_sign = 0  # Reset for switch penalty
        self.hold_steps = 0
        self.last_trade_step = 0
        self.entry_price = 0.0  # Track Entry Price for potential future trade-based rewards
        self.history_net_worth = [self.initial_balance]
        self.episode_returns = deque(maxlen=24)  # For Sortino (last 24 hours)
        
        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback

        self.steps_since_last_trade = 0
        self.trades_in_episode = 0
        self.prev_actions = deque(maxlen=3)
        self.action_history = []
        self.recent_actions = deque(maxlen=5)
        self.recent_position_deltas = deque(maxlen=5)

        self.has_traded_once = False  # Reset exploration bonus

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
            self.data_matrix[:, self.stoch_rsi_norm_idx],
            self.raw_prices,
            window=40, tolerance=8
        )
        bull14, bear14 = self._detect_divergences(
            self.data_matrix[:, self.stoch_14_idx],
            self.raw_prices, window=40, tolerance=8
        )
        bull_rsi, bear_rsi = self._detect_divergences(
            self.data_matrix[:, self.rsi_norm_idx],
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

        # 4. Recurrent Features
        recent_actions_padded = np.array(list(self.recent_actions) + [0.0] * (5 - len(self.recent_actions)), dtype=np.float32)
        recent_deltas_padded = np.array(list(self.recent_position_deltas) + [0.0] * (5 - len(self.recent_position_deltas)), dtype=np.float32)

        full_obs = np.concatenate((std_features, account_features, vp_features, div_vector, recent_actions_padded, recent_deltas_padded))

        # --- DEBUG LOGGING ---
        # Print stats every 10000 steps to avoid spamming, but see what's happening
        if self.current_step % 10000 == 0:
            logging.info(f"\n[DEBUG Step {self.current_step}] Feature Magnitudes:")

            # 1. Check Volume Magnitude
            vol_val = self.data_matrix[self.current_step, self.volume_norm_idx]
            logging.info(f"  > Volume Norm Input:   {vol_val:.5f}  (Should be 0.0 - 1.0)")

            # 2. Check Trend Magnitude (The likely culprit)
            trend_val = self.data_matrix[self.current_step, self.trend_ema_norm_idx]
            logging.info(f"  > Trend EMA Norm Input: {trend_val:.5f}  (Dynamic expanding max -1 to 1)")

            # Debug suggestion:
            current_close = current_price
            ema_val = self.data_matrix[self.current_step, self.ema_50_idx]
            logging.info(f"DEBUG EMA: Close={current_close}, EMA={ema_val}, Diff={current_close - ema_val}")

            # 3. Check Close Pct
            close_pct_val = self.data_matrix[self.current_step, self.close_pct_idx]
            logging.info(f"  > Close Pct Input:     {close_pct_val:.5f}")

            # 4. Check RSI Norm
            rsi_norm_val = self.data_matrix[self.current_step, self.rsi_norm_idx]
            logging.info(f"  > RSI Norm Input:      {rsi_norm_val:.5f}")

            # 5. Check Stoch RSI Norm
            stoch = self.data_matrix[self.current_step, self.stoch_rsi_norm_idx]
            logging.info(f"  > Stoch RSI Norm Input: {stoch:.5f}  (Now -1 to 1)")

            # 6. Check MACD Norm
            macd_norm_val = self.data_matrix[self.current_step, self.macd_norm_idx]
            logging.info(f"  > MACD Norm Input:     {macd_norm_val:.5f}  (Dynamic expanding max -1 to 1)")

            # 7. Check MACD Sig Norm
            macd_sig_norm_val = self.data_matrix[self.current_step, self.macd_sig_norm_idx]
            logging.info(f"  > MACD Sig Norm Input: {macd_sig_norm_val:.5f}  (Dynamic expanding max -1 to 1)")

            # 8. Check ATR Norm
            atr_norm_val = self.data_matrix[self.current_step, self.atr_norm_idx]
            logging.info(f"  > ATR Norm Input:      {atr_norm_val:.5f}")

            # 9. Check Regime
            regime_val = self.data_matrix[self.current_step, self.regime_idx]
            logging.info(f"  > Regime Input:        {regime_val:.5f} (-2 to 2)")

            # 10. Check VP Heatmap Magnitude
            # Access the first available VP day key
            first_day = self.vp_days[0]
            vp_sample = self.vp_data[first_day]['heatmap'][self.current_step]
            logging.info(f"  > VP Heatmap Max:      {np.max(vp_sample):.2f} (Now normalized by sum, max <=1.0)")
            logging.info(f"  > VP Heatmap Values:   {vp_sample}")

            logging.info(f"  > Bull Div Stoch9:    {div_vector[0]:.5f}")
            logging.info(f"  > Bear Div Stoch9:    {div_vector[1]:.5f}")
            logging.info(f"  > Bull Div Stoch14:   {div_vector[2]:.5f}")
            logging.info(f"  > Bear Div Stoch14:   {div_vector[3]:.5f}")
            logging.info(f"  > Bull Div RSI:       {div_vector[4]:.5f}")
            logging.info(f"  > Bear Div RSI:       {div_vector[5]:.5f}")
                        
        return full_obs.astype(np.float32)

    def step(self, action):
        # REVERTED: We keep the raw action magnitude.
        # Values > 1.0 indicate high model confidence.

        self.current_step += 1
        self._sync_wallet_balance()  # Sync balance on each step
        current_price = self.raw_prices[self.current_step]
    
        # === LOOK-AHEAD PnL AVERAGE (this is the magic) ===
        future_returns = []
        for horizon in self.lookahead_horizons:
            if self.current_step + horizon < len(self.df):
                future_price = self.df.iloc[self.current_step + horizon]['close']
                ret = (future_price - current_price) / current_price
                # direction matters → multiply by current position (long = +1, short = -1)
                future_returns.append(ret * self.current_position)

        if future_returns:  # always true except very end of dataset
            num_available = len(future_returns)
            weights = self.lookahead_weights[:num_available]
            lookahead_pnl = np.average(future_returns, weights=weights)
        else:
            lookahead_pnl = 0.0

        # === 3. Optional: still keep a tiny transaction cost so agent doesn’t flip for free ===
        trade_cost = 0.0005 * abs(action[0] - self.current_position)  # 5 bps, very light

        # Final reward = average future PnL minus tiny friction
        reward = lookahead_pnl - trade_cost

        self.prev_shares_held = self.shares_held  # For delta and timely sell bonus

        # Pass RAW action to execution logic
        trade_occurred = self._take_action(action)

        # Extra reward for profitable position reductions (SELLs)
        if trade_occurred and self.shares_held < self.prev_shares_held:
            realized_pnl = (current_price - self.entry_price) * (self.prev_shares_held - self.shares_held)
            if realized_pnl > 0:
                # Small bump; enough for PPO to notice but not dominate PnL
                reward += 0.2

        if trade_occurred:  # Trigger has_traded_once on any trade
            self.has_traded_once = True

        action_val = float(action[0])

        # Hard trade limit penalty (avoid absurd overtrading)
        if self.trades_in_episode > 10:
            reward -= 1.0

        # Log RAW action
        # The chart will now show values outside [-1, 1], preserving information.
        self.prev_actions.append(action_val)
        self.action_history.append(action_val)
        self.recent_actions.append(action_val)
        delta = self.shares_held - self.prev_shares_held
        self.recent_position_deltas.append(delta)

        # Append to returns
        self.returns.append(reward)

        # Compute benchmark return
        benchmark_return = self.data_matrix[self.current_step, self.close_pct_idx]

        # 4. Update Duration Counter
        if trade_occurred:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1

        # Update portfolio returns for Sharpe ratio
        return_step = (self.net_worth - self.prev_net_worth) / (self.prev_net_worth + 1e-8)
        self.portfolio_returns.append(return_step)

        self.prev_net_worth = self.net_worth
        self.prev_shares_held = self.shares_held

        terminated = False
        truncated = False
        if self.current_step >= len(self.df) - 1: truncated = True
        if self.net_worth < (self.initial_balance * 0.5):
            terminated = True
            reward = -10

        obs = self._next_observation()

        self.last_price = current_price
        self.current_position = np.sign(self.shares_held) if abs(self.shares_held) > 1e-6 else 0

        # Info
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
            # Add for debug: "debug_thresholds": {"buy": dynamic_buy_threshold, "sell": dynamic_sell_threshold}
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

        # ---------------------------------------------------------
        # FIX FOR MULTIPROCESSING: Convert Figure to RGB Array
        # ---------------------------------------------------------
        fig.canvas.draw()

        # Convert the canvas buffer to a numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Clear the plot to save memory in the worker
        plt.close(fig)

        return data  # Return the Numpy Array (Safe for SubprocVecEnv)

# Register the environment with Gym
gym.register(
    id='EnhancedTradingEnv-v0',
    entry_point='enhanced_trading_env:EnhancedTradingEnv',
    # Note: df must be provided when creating the env
)