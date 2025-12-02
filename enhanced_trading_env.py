# enhanced_trading_env.py
# Drop-in replacement for trading_env.py — fixes over-trading, adds bonuses, uses Sortino reward
# Compatible with your volume_profile.py and features.py

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

try:
    import wandb
except ImportError:
    wandb = None

from features import get_features


class EnhancedTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        vp_data: Dict[str, dict],
        initial_balance: float = 10000.0,
        lookback_window: int = 100,
        fee_rate: float = 0.0015,
        rsi_bonus_lambda: float = 0.02,
        stoch_bonus_lambda: float = 0.01,
        trade_cooldown_hours: int = 6,
        deadzone: float = 0.15,
        vp_bins: int = 40,
        seed: int = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.original_index = df.index  # Preserve datetime index for VP loc[t]
        self.vp_data = vp_data  # {'vp7': {'poc': array, 'heatmap': array...}, 'vp30': ...}
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.fee_rate = fee_rate
        self.rsi_bonus_lambda = rsi_bonus_lambda
        self.stoch_bonus_lambda = stoch_bonus_lambda
        self.trade_cooldown = trade_cooldown_hours
        self.deadzone = deadzone
        self.vp_bins = vp_bins

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Dynamically compute observation space size
        if len(self.df) > 0:
            sample_step = min(100, len(self.df) - 1)
            end = sample_step + 1
            start = max(0, end - self.lookback_window)
            window = self.df.iloc[start:end].copy()
            window.index = self.original_index[start:end]
            t = window.index[-1]
            vp7_dict = self.vp_data['vp7']
            vp30_dict = self.vp_data['vp30']
            vp7_row = {k: v[sample_step] for k, v in vp7_dict.items()}
            vp30_row = {k: v[sample_step] for k, v in vp30_dict.items()}
            vp7_df = pd.DataFrame([vp7_row], index=[t])
            vp30_df = pd.DataFrame([vp30_row], index=[t])
            sample_features = get_features(window, vp7_df, vp30_df, t)
            feat_len = len(sample_features)
        else:
            feat_len = 139  # Fallback

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feat_len,), dtype=np.float32)

        self.seed(seed)
        self.phase = 1

        # State
        self.last_obs = None
        self.last_trade_step = -9999
        self.net_worth_history = []

        self.reset(seed=seed)

    def set_phase(self, new_phase: int):
        self.phase = new_phase

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.last_trade_step = -9999
        self.net_worth_history = [self.initial_balance]

        # Random episode: 30-90 days, adjusted for data length
        episode_len = self.np_random.integers(720, min(2161, len(self.df)))
        max_start = len(self.df) - episode_len
        if max_start < 0:
            max_start = 0
            episode_len = len(self.df)
        self.current_step = self.np_random.integers(0, max_start + 1)

        obs = self._get_observation()
        self.last_obs = obs.copy()
        return obs, {}

    def _get_observation(self):
        end = self.current_step + 1
        start = max(0, end - self.lookback_window)
        window = self.df.iloc[start:end].copy()
        window.index = self.original_index[start:end]  # Restore datetime index for t

        t = window.index[-1]  # Timestamp for VP loc[t]

        # Extract current row from vp_data arrays
        vp7_dict = self.vp_data['vp7']
        vp30_dict = self.vp_data['vp30']
        vp7_row = {k: v[self.current_step] for k, v in vp7_dict.items()}
        vp30_row = {k: v[self.current_step] for k, v in vp30_dict.items()}

        # Build single-row DataFrames for get_features (expects df.loc[t])
        vp7_df = pd.DataFrame([vp7_row], index=[t])
        vp30_df = pd.DataFrame([vp30_row], index=[t])

        features = get_features(window, vp7_df, vp30_df, t)
        obs = np.array(features, dtype=np.float32)
        return obs

    def step(self, action: np.ndarray):
        action_val = float(action[0])
        price = float(self.df.iloc[self.current_step]["close"])
        timestamp = self.original_index[self.current_step]

        info = {}
        reward = 0.0

        # Cooldown + deadzone
        hours_since_trade = self.current_step - self.last_trade_step
        if hours_since_trade < self.trade_cooldown or abs(action_val) < self.deadzone:
            action_val = 0.0

        trade_executed = False
        if abs(action_val) >= self.deadzone and hours_since_trade >= self.trade_cooldown:
            target = np.clip(action_val, -1.0, 1.0)
            target_position_value = target * self.net_worth
            target_shares = target_position_value / price if price > 0 else 0.0
            delta_shares = target_shares - self.shares_held
            trade_value = delta_shares * price
            fee = abs(trade_value) * self.fee_rate
            self.balance += -trade_value - fee
            self.shares_held = target_shares
            self.last_trade_step = self.current_step
            self.trade_count += 1
            trade_executed = True
            info["trade"] = True
            info["fee"] = fee

        # Update net worth every step (includes hold PnL)
        self.net_worth = self.balance + self.shares_held * price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        self.net_worth_history.append(self.net_worth)
        self.prev_net_worth = self.net_worth

        # FIXED REWARD: Sortino (risk-adjusted) + bonuses
        if len(self.net_worth_history) > 10:
            recent_len = min(21, len(self.net_worth_history))
            recent_history = self.net_worth_history[-recent_len:]
            returns = np.diff(recent_history) / recent_history[:-1]
            mean_ret = np.mean(returns)
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-6
            sortino = mean_ret / downside_std if downside_std > 0 else 0.0
            reward = sortino * 30.0  # Scale for SAC
        else:
            reward = 0.0

        # RSI/Stoch bonuses (mean-reversion nudge) - using approximate indices, normalize rsi
        rsi_bonus = stoch_bonus = 0.0
        if self.last_obs is not None and trade_executed:
            rsi_raw = self.last_obs[111]
            rsi_norm = min(rsi_raw / 100.0, 1.0) if rsi_raw > 0 else 0.5
            stoch_norm = np.clip(self.last_obs[112], 0.0, 1.0)
            rsi_bonus = self.rsi_bonus_lambda * (1.0 - abs(rsi_norm - 0.5))
            stoch_bonus = self.stoch_bonus_lambda * (1.0 - abs(stoch_norm - 0.5))
        reward += rsi_bonus + stoch_bonus

        # VP proximity bonus (trade near POC)
        vp_bonus = 0.0
        if 'vp7' in self.vp_data:
            vp7 = self.vp_data['vp7']
            if self.current_step < len(vp7['poc']):
                poc = vp7['poc'][self.current_step]
                if not np.isnan(poc) and poc > 0:
                    dist = abs(price - poc) / price
                    vp_bonus = 0.08 * (1.0 - min(dist / 0.015, 1.0))  # +0.08 max if <1.5%
        reward += vp_bonus
        
        if self.current_step % 1500 == 0:
            current_date = timestamp
            poc = self.vp_data['vp7']['poc'][self.current_step]
            vah = self.vp_data['vp7']['vah'][self.current_step]
            val = self.vp_data['vp7']['val'][self.current_step]
            heatmap = self.vp_data['vp7']['heatmap'][self.current_step]  # Always vp_bins length

            dist_pct = ((price - poc) / poc) * 100 if poc != 0 else 0
            print(f"Step {self.current_step} [{current_date}]: P={price:.0f} | POC={poc:.0f} ({dist_pct:+.2f}%) | NetWorth={self.net_worth:.0f} | ATH={self.max_net_worth:.0f}")
        # Advance
        self.current_step += 1
        terminated = self.net_worth < self.initial_balance * 0.5
        truncated = self.current_step >= len(self.df) - 1

        obs = self._get_observation()
        self.last_obs = obs.copy()

        # VP heatmap for current step (before advance)
        vp_heatmap = np.zeros(self.vp_bins)
        if 'vp7' in self.vp_data:
            vp7 = self.vp_data['vp7']
            if self.current_step - 1 < len(vp7['heatmap']):
                vp_heatmap = vp7['heatmap'][self.current_step - 1]

        info.update({
            "portfolio_value": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_val,
            "reward": reward,
            "price": price,
            "date": timestamp,
            "vp_heatmap": vp_heatmap,
            "rsi_bonus": rsi_bonus,
            "stoch_bonus": stoch_bonus,
            "vp_bonus": vp_bonus,
        })
        if wandb and wandb.run:
            wandb.log(info)

        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | NW: ${self.net_worth:,.0f} | Shares: {self.shares_held:.3f}")

    def close(self):
        pass