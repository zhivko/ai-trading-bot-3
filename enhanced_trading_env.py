# enhanced_trading_env.py
# Drop this file in your repo root (next to trading_env.py)
# Then in main.py: from enhanced_trading_env import EnhancedTradingEnv as TradingEnv

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
        vp_data: Dict[str, pd.DataFrame],
        initial_balance: float = 1000.0,
        fee_rate: float = 0.0015,
        rsi_bonus_lambda: float = 0.02,
        stoch_bonus_lambda: float = 0.01,
        trade_cooldown_hours: int = 6,
        deadzone: float = 0.15,
        seed: int = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.vp_data = vp_data
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.rsi_bonus_lambda = rsi_bonus_lambda
        self.stoch_bonus_lambda = stoch_bonus_lambda
        self.trade_cooldown = trade_cooldown_hours
        self.deadzone = deadzone

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(259,), dtype=np.float32)  # adjust if you changed features

        self.seed(seed)

        # Critical state
        self.last_obs = None
        self.phase = 1
        self.last_trade_step = -9999
        self.net_worth_history = []

        self.reset(seed=seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0
        self.entry_price = None
        self.last_trade_step = -9999
        self.net_worth_history = [self.initial_balance]

        # Random episode 30–90 days
        episode_len = self.np_random.integers(720, 2161)
        max_start = len(self.df) - episode_len
        self.current_step = self.np_random.integers(0, max_start + 1)

        obs = self._get_observation()
        self.last_obs = obs.copy()
        return obs, {}

    def _get_observation(self):
        # Preserve original timestamps so vp7_df.loc[t] works
        end = self.current_step + 1
        start = max(0, end - 100)
        window = self.df.iloc[start:end].copy()
        window.index = self.df.index[start:end]  # ← CRITICAL: restores datetime index

        features = get_features(window, self.vp_data)
        obs = np.array(features, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def step(self, action: np.ndarray):
        action_val = float(action[0])
        price = float(self.df.iloc[self.current_step]["close"])
        timestamp = self.df.index[self.current_step]

        info = {}
        reward = 0.0

        # === 1. Trade cooldown & deadzone (kills over-trading) ===
        hours_since_trade = (self.current_step - self.last_trade_step)
        can_trade = hours_since_trade >= self.trade_cooldown
        abs_action = abs(action_val)

        if not can_trade or abs_action < self.deadzone:
            action_val = 0.0  # Force flat

        # === 2. Execute trade ===
        if abs_action >= self.deadzone and can_trade:
            target = np.clip(action_val, -1.0, 1.0)
            if self.position != target:
                # Fee only on position change
                trade_size = abs(target - self.position)
                fee = trade_size * self.net_worth * self.fee_rate
                self.balance -= fee
                self.position = target
                self.entry_price = price
                self.last_trade_step = self.current_step
                info["trade"] = True
                info["fee"] = fee

        # === 3. Net worth ===
        if self.position != 0 and self.entry_price is not None:
            pnl_factor = self.position * (price / self.entry_price - 1)
            self.net_worth = self.balance * (1 + pnl_factor)
        else:
            self.net_worth = self.balance
        self.net_worth_history.append(self.net_worth)

        # === 4. Base reward (PnL) ===
        total_pct = (self.net_worth / self.initial_balance - 1)
        reward = total_pct * 100

        # === 5. RSI + Stoch mean-reversion bonus (only on real entries) ===
        rsi_bonus = stoch_bonus = 0.0
        if self.last_obs is not None and abs_action >= self.deadzone and can_trade:
            rsi = self.last_obs[0]
            stoch = self.last_obs[1]
            rsi_bonus = self.rsi_bonus_lambda * (1.0 - abs(rsi - 0.5))
            stoch_bonus = self.stoch_bonus_lambda * (1.0 - abs(stoch - 0.5))
        reward += rsi_bonus + stoch_bonus

        # === 6. VP proximity bonus (encourage trading near POC) ===
        vp_bonus = 0.0
        if 'vp7' in self.vp_data:
            vp7_row = self.vp_data['vp7']
            if timestamp in vp7_row.index:
                poc = vp7_row.loc[timestamp, 'poc']
                if pd.notna(poc):
                    dist = abs(price - poc) / price
                    vp_bonus = 0.08 * (1.0 - min(dist / 0.02, 1.0))  # max +0.08 if within 2%
        reward += vp_bonus

        # === 7. Advance ===
        self.current_step += 1
        done = (
            self.current_step >= len(self.df) - 1
            or self.net_worth < self.initial_balance * 0.5
        )

        obs = self._get_observation()
        self.last_obs = obs.copy()

        info.update({
            "net_worth": self.net_worth,
            "position": self.position,
            "rsi_bonus": rsi_bonus,
            "stoch_bonus": stoch_bonus,
            "vp_bonus": vp_bonus,
            "total_pct": total_pct * 100,
        })

        if wandb and wandb.run:
            wandb.log(info)

        return obs, float(reward), bool(done), bool(done), info

    def render(self):
        pass

    def close(self):
        pass