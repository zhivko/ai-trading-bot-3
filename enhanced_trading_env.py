import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from volume_profile import get_rolling_vp
import logging


class EnhancedTradingEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'render_fps': 4,
    }

    def __init__(
        self,
        df,
        initial_balance=10000,
        lookback_window=50,
        vp_days=None,
        vp_bins=40,
        buy_threshold=0.5,
        sell_threshold=-0.5,
        precalculated_vp=None,
        trading_fee_multiplier=0.00075,
        phase=1,
        min_trade_value_usd=10.0,
        pair='BTCUSDT',
        timeframe='1h',
        split_date=None
    ):
        super(EnhancedTradingEnv, self).__init__()
        logging.info("Initializing EnhancedTradingEnv...")
        self.metadata.update({
            'pair': pair,
            'timeframe': timeframe,
            'split_date': split_date,
        })

        # Config
        self.initial_balance = initial_balance
        self.transaction_cost_rate = 0.0015
        self.lookback_window = lookback_window
        self.vp_days = vp_days if vp_days else [7, 30]
        self.vp_bins = vp_bins
        self.trading_fee_multiplier = trading_fee_multiplier
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.phase = phase
        self.min_trade_value_usd = float(min_trade_value_usd)
        self.pair = pair
        self.timeframe = timeframe

        self.lookahead_horizons = [1, 4, 12, 24]
        self.lookahead_weights = [0.4, 0.3, 0.2, 0.1]

        self.max_leverage = 5.0
        self.leverage_buffer = 0.95

        # Data
        self.raw_df = df.reset_index(drop=False)
        self.df = self.raw_df.copy()
        self.df = self.df.drop(columns=['timestamp'], errors='ignore')

        # Volume Profile
        if precalculated_vp:
            self.vp_data = precalculated_vp
        else:
            self.vp_data = {}
            for days in self.vp_days:
                self.vp_data[days] = get_rolling_vp(self.raw_df, days, bins=self.vp_bins)

        # === FEATURE ENGINEERING ===
        self.df['close_pct'] = self.df['close'].pct_change().fillna(0)

        # Volume normalization
        vol_rolling_max = self.df['volume'].rolling(window=100, min_periods=1).max()
        self.df['volume_norm'] = self.df['volume'] / (vol_rolling_max + 1e-8)
        self.df['volume_norm'] = self.df['volume_norm'].fillna(0)

        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi_norm'] = (self.df['rsi'] - 50.0) / 50.0

        # Stoch RSI
        min_rsi = self.df['rsi'].rolling(14).min()
        max_rsi = self.df['rsi'].rolling(14).max()
        self.df['stoch_rsi'] = (self.df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-8)
        self.df['stoch_rsi_norm'] = (self.df['stoch_rsi'] - 0.5) / 0.5
        self.df['stoch_rsi_norm'] = self.df['stoch_rsi_norm'].fillna(0)

        # MACD
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        macd_max = self.df['macd'].abs().expanding(min_periods=200).max().replace(0, 1)
        self.df['macd_norm'] = self.df['macd'] / macd_max
        self.df['macd_sig_norm'] = self.df['macd_signal'] / macd_max.replace(0, 1)

        # EMA50 + Trend
        self.df['ema_50'] = self.df['close'].ewm(span=50, adjust=False).mean()
        dist = self.df['close'] - self.df['ema_50']
        max_dist = dist.abs().expanding(min_periods=200).max().fillna(self.df['close'] * 0.01)
        self.df['trend_ema_norm'] = dist / max_dist.replace(0, 1)

        # ATR
        tr = np.maximum.reduce([
            self.df['high'] - self.df['low'],
            np.abs(self.df['high'] - self.df['close'].shift()),
            np.abs(self.df['low'] - self.df['close'].shift())
        ])
        self.df['atr'] = pd.Series(tr).rolling(14).mean().fillna(0.01)
        self.df['atr_norm'] = self.df['atr'] / self.df['close']

        # Regime
        self.df['regime'] = self.df['trend_ema_norm'] / (self.df['atr_norm'] + 1e-8)
        self.df['regime'] = np.clip(self.df['regime'], -2, 2)

        # Stochastic 14
        low14 = self.df['low'].rolling(14).min()
        high14 = self.df['high'].rolling(14).max()
        self.df['stoch_14'] = (self.df['close'] - low14) / (high14 - low14 + 1e-8)
        self.df['stoch_14'] = self.df['stoch_14'].fillna(0.5).clip(0, 1)

        # Final feature list
        self.features = [
            'close_pct', 'volume_norm', 'rsi_norm', 'stoch_rsi_norm',
            'macd_norm', 'macd_sig_norm', 'trend_ema_norm', 'atr_norm', 'regime'
        ]
        self.div_features = [
            'bull_div_stoch9', 'bear_div_stoch9',
            'bull_div_stoch14', 'bear_div_stoch14',
            'bull_div_rsi', 'bear_div_rsi'
        ]
        self.div_scores = {k: 0.0 for k in self.div_features}

        self.df.fillna(0, inplace=True)
        self.data_matrix = self.df.values.astype(np.float32)

        # Fast column access
        self.close_idx = self.df.columns.get_loc('close')
        self.raw_prices = self.data_matrix[:, self.close_idx]
        feature_idx = [self.df.columns.get_loc(f) for f in self.features]
        self.market_features = self.data_matrix[:, feature_idx]

        # Spaces
        market_size = self.lookback_window * len(self.features)
        vp_size = len(self.vp_days) * (3 + self.vp_bins)
        total_size = market_size + 2 + vp_size + len(self.div_features) + 10

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)

        # State tracking
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.max_lookback = max(max(d * 24 for d in self.vp_days), 30) + self.lookback_window

        self.recent_actions = deque(maxlen=5)
        self.recent_position_deltas = deque(maxlen=5)
        self.action_history = []
        self.history_net_worth = []

        # Index caches
        self.stoch_rsi_norm_idx = self.df.columns.get_loc('stoch_rsi_norm')
        self.stoch_14_idx = self.df.columns.get_loc('stoch_14')
        self.rsi_norm_idx = self.df.columns.get_loc('rsi_norm')

        # Feature names (for saliency)
        self.feature_names = self.get_feature_names()
        logging.info("Initializing EnhancedTradingEnv...Done.")


    # ===================================================================
    # Helper Methods
    # ===================================================================

    def get_feature_names(self):
        names = self.features.copy()
        names += ["balance_norm", "holdings_norm"]
        for day in self.vp_days:
            p = f"vp_{day}d"
            names += [f"{p}_dist_poc", f"{p}_dist_vah", f"{p}_dist_val"]
            names += [f"{p}_bucket_{i}" for i in range(self.vp_bins)]
        names += self.div_features
        names += [f"recent_action_{i}" for i in range(5)]
        names += [f"position_delta_{i}" for i in range(5)]
        return names

    def _detect_divergences(self, series, price_series, window=40, tolerance=8):
        if self.current_step < window + 20:
            return 0.0, 0.0
        start = self.current_step - window
        prices = price_series[start:self.current_step + 1]
        osc = series[start:self.current_step + 1]
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

    def _take_action(self, action):
        current_price = self.raw_prices[self.current_step]
        action_val = float(action[0])

        target_usd = action_val * self.net_worth * self.max_leverage * self.leverage_buffer
        current_usd = self.shares_held * current_price
        trade_usd = target_usd - current_usd

        if abs(trade_usd) < self.min_trade_value_usd:
            return False

        shares_to_trade = trade_usd / current_price
        cost = abs(shares_to_trade * current_price) * self.transaction_cost_rate

        if trade_usd > 0:  # Buy
            if self.balance >= shares_to_trade * current_price + cost:
                self.shares_held += shares_to_trade
                self.balance -= shares_to_trade * current_price + cost
                if self.shares_held > 0 and self.entry_price == 0.0:   # first long entry
                    self.entry_price = current_price
                return True
        else:  # Sell / cover
            if self.shares_held >= -shares_to_trade:
                self.shares_held += shares_to_trade
                self.balance += abs(shares_to_trade * current_price) - cost
                return True
        return False

    # ===================================================================
    # Core Gym Methods
    # ===================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.shares_held = 0.0
        self.current_position = 0
        self.entry_price = 0.0                    # ← THIS WAS MISSING
        self.history_net_worth = [self.initial_balance]
        self.recent_actions.clear()
        self.recent_position_deltas.clear()
        self.action_history = []
        self.trades_in_episode = 0
        self.has_traded_once = False

        if len(self.df) > self.max_lookback + 1000:
            self.current_step = self.np_random.integers(self.max_lookback, len(self.df) - 1000)
        else:
            self.current_step = self.max_lookback

        return self._next_observation(), {}

    def _next_observation(self):
        start = self.current_step - self.lookback_window
        market = self.market_features[start:self.current_step].flatten()

        price = self.raw_prices[self.current_step]
        balance_norm = self.balance / self.initial_balance
        holdings_norm = (self.shares_held * price) / self.initial_balance
        account = np.array([balance_norm, holdings_norm], dtype=np.float32)

        vp_parts = []
        for days in self.vp_days:
            d = self.vp_data[days]
            poc = d['poc'][self.current_step]
            vah = d['vah'][self.current_step]
            val = d['val'][self.current_step]
            heatmap = d['heatmap'][self.current_step].astype(np.float32)
            total = heatmap.sum()
            if total > 0:
                heatmap /= total

            dist_poc = (poc - price) / price if price > 0 and poc > 0 else 0
            dist_vah = (vah - price) / price if price > 0 else 0
            dist_val = (val - price) / price if price > 0 else 0

            vp_parts.extend([dist_poc, dist_vah, dist_val])
            vp_parts.extend(heatmap)
        vp_vec = np.array(vp_parts, dtype=np.float32)

        # Divergences
        for k in self.div_scores:
            self.div_scores[k] *= 0.95
        b9, br9 = self._detect_divergences(self.data_matrix[:, self.stoch_rsi_norm_idx], self.raw_prices)
        b14, br14 = self._detect_divergences(self.data_matrix[:, self.stoch_14_idx], self.raw_prices)
        br, brr = self._detect_divergences(self.data_matrix[:, self.rsi_norm_idx], self.raw_prices)
        if b9: self.div_scores['bull_div_stoch9'] = 1.0
        if br9: self.div_scores['bear_div_stoch9'] = 1.0
        if b14: self.div_scores['bull_div_stoch14'] = 1.0
        if br14: self.div_scores['bear_div_stoch14'] = 1.0
        if br: self.div_scores['bull_div_rsi'] = 1.0
        if brr: self.div_scores['bear_div_rsi'] = 1.0
        div_vec = np.array([self.div_scores[k] for k in self.div_features], dtype=np.float32)

        # Recurrent
        act_pad = np.array(list(self.recent_actions) + [0.] * (5 - len(self.recent_actions)), dtype=np.float32)
        delta_pad = np.array(list(self.recent_position_deltas) + [0.] * (5 - len(self.recent_position_deltas)), dtype=np.float32)

        obs = np.concatenate([market, account, vp_vec, div_vec, act_pad, delta_pad])
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        current_price = self.raw_prices[self.current_step]

        # Look-ahead reward
        future_rets = []
        for h, w in zip(self.lookahead_horizons, self.lookahead_weights):
            if self.current_step + h < len(self.df):
                fp = self.df.iloc[self.current_step + h]['close']
                ret = (fp - current_price) / current_price * np.sign(self.shares_held)
                future_rets.append(ret)
        lookahead = np.average(future_rets, weights=self.lookahead_weights[:len(future_rets)]) if future_rets else 0

        trade_cost = 0.0005 * abs(action[0] - self.current_position)
        reward = lookahead - trade_cost

        prev_shares = self.shares_held
        trade_happened = self._take_action(action)

        if trade_happened and self.shares_held < prev_shares:
            realized = (current_price - self.entry_price) * (prev_shares - self.shares_held)
            if realized > 0:
                reward += 0.2

        action_val = float(action[0])
        self.recent_actions.append(action_val)
        self.recent_position_deltas.append(self.shares_held - prev_shares)
        self.action_history.append(action_val)

        self.net_worth = self.balance + self.shares_held * current_price
        self.history_net_worth.append(self.net_worth)

        terminated = self.net_worth <= self.initial_balance * 0.5
        truncated = self.current_step >= len(self.df) - 1

        obs = self._next_observation()
        info = {"net_worth": self.net_worth, "price": current_price, "action": action_val}

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