# enhanced_trading_env.py
# Drop this file in your repo root (next to trading_env.py)
# Then in main.py: from enhanced_trading_env import EnhancedTradingEnv as TradingEnv

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

try:
    import wandb
except ImportError:
    wandb = None

from features import get_features
from volume_profile import get_rolling_vp


class EnhancedTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        vp_days: List[int],
        initial_balance: float = 1000.0,
        fee_rate: float = 0.0015,
        rsi_bonus_lambda: float = 0.02,
        stoch_bonus_lambda: float = 0.01,
        trade_cooldown_hours: int = 6,
        deadzone: float = 0.15,
        seed: int = None,
    ):
        super().__init__()

        self.timestamps = df.index

        self.raw_df = df.reset_index(drop=False) 
        self.df = self.raw_df.copy()

        # Compute vp_data internally
        vp_data = {}
        for days in vp_days:
            vp = get_rolling_vp(df, window_days=days, num_bins=40)
            vp_df = pd.DataFrame(index=df.index)
            vp_df['poc'] = vp['poc']
            vp_df['vah'] = vp['vah']
            vp_df['val'] = vp['val']
            vp_df['heatmap'] = list(vp['heatmap'])
            vp_df['hvn'] = vp['hvn']
            vp_df['lvn'] = vp['lvn']
            vp_data[f'vp{days}'] = vp_df
        self.vp_data = vp_data

        # Add volume profile columns to raw_df for logging access
        if 'vp7' in self.vp_data:
            self.raw_df['poc'] = self.vp_data['vp7']['poc'].values
            self.raw_df['vah'] = self.vp_data['vp7']['vah'].values
            self.raw_df['val'] = self.vp_data['vp7']['val'].values
            self.raw_df['heatmap'] = self.vp_data['vp7']['heatmap'].values
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.rsi_bonus_lambda = rsi_bonus_lambda
        self.stoch_bonus_lambda = stoch_bonus_lambda
        self.trade_cooldown = trade_cooldown_hours
        self.deadzone = deadzone
        self.raw_prices = self.df['close'].values.astype(np.float32)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(139,), dtype=np.float32)  # adjust if you changed features

        self.seed(seed)

        # Critical state
        self.last_obs = None
        self.phase = 1
        self.last_trade_step = -9999
        self.net_worth_history = []
        self.max_net_worth = self.initial_balance

        self.reset(seed=seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def set_phase(self, new_phase):
        self.phase = new_phase


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
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
        window.index = self.timestamps[start:end]  # ← CRITICAL: restores datetime index

        features = get_features(window, self.vp_data['vp7'], self.vp_data['vp30'], self.timestamps[self.current_step])
        obs = np.array(features, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def step(self, action: np.ndarray):
        action_val = float(action[0])
        current_price = self.raw_prices[self.current_step]
        timestamp = self.timestamps[self.current_step]

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
                self.entry_price = current_price
                self.last_trade_step = self.current_step
                info["trade"] = True
                info["fee"] = fee

        # === 3. Net worth ===
        if self.position != 0 and self.entry_price is not None:
            pnl_factor = self.position * (current_price / self.entry_price - 1)
            self.net_worth = self.balance * (1 + pnl_factor)
        else:
            self.net_worth = self.balance
        self.net_worth_history.append(self.net_worth)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

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
                    dist = abs(current_price - poc) / current_price
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

        if self.current_step % 1500 == 0:
            current_date = self.timestamps[self.current_step]
            poc = self.raw_df['poc'][self.current_step]
            vah = self.raw_df['vah'][self.current_step]
            val = self.raw_df['val'][self.current_step]
            heatmap = self.raw_df['heatmap'][self.current_step] # Always vp_bins length

            vah = self.raw_df['vah'][self.current_step]

            dist_pct = ((current_price - poc) / poc) * 100 if poc != 0 else 0
            print(f"Step {self.current_step} [{current_date}]: P={current_price:.0f} | POC={poc:.0f} ({dist_pct:+.2f}%) | NetWorth={self.net_worth:.0f} | ATH={self.max_net_worth:.0f}")


        return obs, float(reward), bool(done), bool(done), info

    def render(self):
        pass

    def close(self):
        pass