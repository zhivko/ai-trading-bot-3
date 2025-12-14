import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import talib

class ContinuousTradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000, window_size=20, commission=0.001):
        super(ContinuousTradingEnv, self).__init__()

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.episode_trade_count = 0  # Counter for trades in current episode
        
        # --- FIX 1: Separate data for plotting/trading vs. observation features ---
        # 1. raw_data_for_plot: Retain absolute OHLCV and timestamp for trading and plotting.
        #    This data is used in reset(), step(), and render().
        self.raw_data_for_plot = self._clean_data_for_plot(df)
        
        # 2. self.df: Contains the normalized features (used by _get_observation for the agent).
        self.df = self._create_features(self.raw_data_for_plot.copy())
        
        # --- NEW: History logging structure ---
        self.history = self._get_history_template()

        # --- Observation Space ---
        # The agent observes a window_size of normalized market data (12 features: O,H,L,C,V,EMA, MACD, RSI, STO_RSI)
        # plus 2 internal state features (Current Balance, Current Holding Amount).
        obs_shape = (window_size * 12 + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # --- Action Space (Continuous for SAC/PPO) ---
        # Action is a single float between -1.0 (Strong Sell) and 1.0 (Strong Buy)
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        
        self.reset()

    def _clean_data_for_plot(self, df):
        """Prepares raw data by cleaning and setting timestamp."""
        df_cleaned = df.copy()
        df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'])
        df_cleaned = df_cleaned.set_index('timestamp')
        df_cleaned = df_cleaned.dropna()
        # Reset index to a numerical range (0, 1, 2...) for easy slicing with self.current_step
        return df_cleaned.reset_index()

    def _create_features(self, df_features):
        """Creates normalized features for the agent's observation space."""
        # Calculate indicators using TA-Lib if not already present
        if 'ema_50' not in df_features.columns:
            df_features['ema_50'] = talib.EMA(df_features['close'].values, timeperiod=50)
            df_features['ema_200'] = talib.EMA(df_features['close'].values, timeperiod=200)
            macd, macdsignal, macdhist = talib.MACD(df_features['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            df_features['macd'] = macd
            df_features['macd_signal'] = macdsignal
            df_features['rsi'] = talib.RSI(df_features['close'].values, timeperiod=14)
            fastk_3_14, fastd_3_14 = talib.STOCHRSI(df_features['close'].values, timeperiod=14, fastk_period=3, fastd_period=3)
            df_features['sto_rsi_3_14'] = fastk_3_14
            fastk_10_60, fastd_10_60 = talib.STOCHRSI(df_features['close'].values, timeperiod=60, fastk_period=10, fastd_period=3)
            df_features['sto_rsi_10_60'] = fastk_10_60
        # 2. Normalize Price Data and EMA (for observation features)
        price_cols = ['open', 'high', 'low', 'close', 'ema_50', 'ema_200']
        for col in price_cols:
            # Replaces absolute price with percentage change
            df_features.loc[:, col] = df_features[col].pct_change()
            
        # Normalize volume
        df_features.loc[:, 'volume'] = df_features['volume'] / df_features['volume'].rolling(self.window_size).mean()
        
        # Normalize MACD indicators (already relative values, normalize by rolling mean)
        df_features.loc[:, 'macd'] = df_features['macd'] / df_features['macd'].rolling(self.window_size).std().fillna(1)
        df_features.loc[:, 'macd_signal'] = df_features['macd_signal'] / df_features['macd_signal'].rolling(self.window_size).std().fillna(1)
        
        # RSI and Stochastic RSI are already bounded (0-100), normalize to 0-1 range
        df_features.loc[:, 'rsi'] = df_features['rsi'] / 100.0
        df_features.loc[:, 'sto_rsi_3_14'] = df_features['sto_rsi_3_14'] / 100.0
        df_features.loc[:, 'sto_rsi_10_60'] = df_features['sto_rsi_10_60'] / 100.0
        
        df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)
        # Drop the 'timestamp' column if it exists
        if 'timestamp' in df_features.columns:
            df_features = df_features.drop(columns=['timestamp'])
        return df_features

    def _get_history_template(self):
        """Creates the dictionary structure for logging episode history."""
        return {
            'date': [],
            'net_worth': [],
            'action': [],
            'trade_type': [], # 'buy', 'sell', 'hold'
            'trade_price': [],
        }

    def _get_observation(self):
        # Extract features for the current time step window
        window_data = self.df.loc[self.current_step - self.window_size + 1:self.current_step,
                                  ['open', 'high', 'low', 'close', 'volume', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60']].values.flatten()
        
        # Append internal state (Normalized Balance and Holding)
        state_features = np.array([
            self.balance / self.initial_balance,
            self.shares_held / self.max_shares
        ])
        
        return np.concatenate((window_data, state_features))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.episode_trade_count = 0  # Reset trade counter for new episode
        
        # Start at a random point after the first observation window
        self.current_step = self.window_size + self.np_random.integers(len(self.df) - self.window_size - 100)

        # FIX 2: Use raw data for correct price calculation
        initial_price = self.raw_data_for_plot.loc[self.current_step]['close']
        self.max_shares = self.initial_balance / initial_price

        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance

        # New: Reset and initialize history for the new episode
        self.history = self._get_history_template()
        # FIX 3: Correctly log the datetime value
        self.history['date'].append(self.raw_data_for_plot.loc[self.current_step, 'timestamp'])
        self.history['net_worth'].append(self.net_worth)
        self.history['action'].append(0.0)
        self.history['trade_type'].append('hold')
        self.history['trade_price'].append(np.nan)

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Action is a single float in [-1, 1]. Positive is Buy, Negative is Sell.
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            terminated = True
        else:
            terminated = False
        
        # FIX 4: Use raw data for correct price calculation
        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        action_magnitude = action[0] if hasattr(action, '__len__') else action
        
        # --- Execute Trade based on Action ---
        if action_magnitude > 0: # Buy
            # Agent tries to allocate a percentage of current cash based on magnitude
            cash_to_spend_ratio = action_magnitude
            max_can_buy = self.balance / (current_price * (1 + self.commission)) # Factor in commission for max buy

            shares_to_buy = max_can_buy * cash_to_spend_ratio
            trade_cost = shares_to_buy * current_price * (1 + self.commission)

            if trade_cost <= self.balance:
                self.shares_held += shares_to_buy
                self.balance -= trade_cost

        elif action_magnitude < 0: # Sell
            # Agent tries to sell a percentage of current holdings based on magnitude
            shares_to_sell_ratio = abs(action_magnitude)

            shares_to_sell = self.shares_held * shares_to_sell_ratio
            trade_revenue = shares_to_sell * current_price * (1 - self.commission)

            self.shares_held -= shares_to_sell
            self.balance += trade_revenue

        # Determine if a trade actually occurred for logging and penalties
        trade_type = 'hold'
        trade_price_log = np.nan

        if 'shares_to_buy' in locals() and shares_to_buy > 0:
            trade_type = 'buy'
            trade_price_log = current_price
        elif 'shares_to_sell' in locals() and shares_to_sell > 0:
            trade_type = 'sell'
            trade_price_log = current_price

        # Set trade_executed flag for callback
        trade_executed = trade_type != 'hold'

        # Increment trade counter and apply frequency penalty
        frequency_penalty = 0  # Initialize to avoid UnboundLocalError
        if trade_executed:
            self.episode_trade_count += 1
            # Frequency penalty: increases with each trade (quadratic penalty for overtrading)
            frequency_penalty = -0.001 * (self.episode_trade_count ** 2)

        # --- Calculate Reward and Update State (The Critical Fix) ---
        self.net_worth = self.balance + self.shares_held * current_price

        # 1. Base Reward: Use Log Return for smoother, more stable training
        net_worth_ratio = self.net_worth / self.previous_net_worth
        # Use np.log(max(..., 1e-6)) to avoid log of zero/negative
        base_reward = np.log(max(net_worth_ratio, 1e-6))

        # 2. Action Penalty: Strongly penalize any trade action to reduce over-trading
        TRADE_PENALTY_COEF = 0.005 # 0.5% penalty for any action magnitude

        # The penalty scales with how large the action was
        action_penalty = -TRADE_PENALTY_COEF * abs(action_magnitude)

        # 3. Bankruptcy Check: Terminate episode if net worth falls below 30% of initial balance
        if self.net_worth < (self.initial_balance * 0.3):
            terminated = True
            reward = -10  # Large penalty for bankruptcy
        else:
            # 4. Final Reward (only if not bankrupt)
            reward = base_reward + action_penalty + frequency_penalty

        self.previous_net_worth = self.net_worth

        observation = self._get_observation()

        # FIX 5: Correctly log the datetime value
        self.history['date'].append(self.raw_data_for_plot.loc[self.current_step, 'timestamp'])
        self.history['net_worth'].append(self.net_worth)
        self.history['action'].append(float(action_magnitude))
        self.history['trade_type'].append(trade_type)
        self.history['trade_price'].append(trade_price_log)

        info = {
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'current_price': current_price,
            'action': action_magnitude,
            'timestamp': self.raw_data_for_plot.loc[self.current_step, 'timestamp'],
            'trade_executed': trade_executed
        }

        return observation, reward, terminated, False, info

    def render(self, mode='human', agent_name='RL_Agent'):
        """Plots the agent's performance using mplfinance, saving the result to a file."""
        if mode == 'human':
            # 1. Convert history to DataFrame and set a DatetimeIndex
            history_df = pd.DataFrame(self.history)
            history_df.set_index(pd.to_datetime(history_df['date']), inplace=True)

            # Slice the raw_data_for_plot for the episode duration
            start_idx = self.current_step - len(history_df) + 1
            end_idx = self.current_step + 1
            
            episode_df_raw = self.raw_data_for_plot.iloc[start_idx:end_idx].copy()
            episode_df = episode_df_raw.set_index('timestamp').drop(columns=['ema_50', 'ema_200', 'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60'], errors='ignore')
            episode_df.index.name = 'Date'

            # --- 2. Prepare Trades for mplfinance scatter plot ---
            buys = history_df[history_df['trade_type'] == 'buy']
            sells = history_df[history_df['trade_type'] == 'sell']

            buy_prices = pd.Series(index=episode_df.index, data=np.nan)
            sell_prices = pd.Series(index=episode_df.index, data=np.nan)
            buy_prices.loc[buys.index] = buys['trade_price']
            sell_prices.loc[sells.index] = sells['trade_price']

            buy_plots = mpf.make_addplot(
                buy_prices, type='scatter', markersize=100, marker='^', color='green', label='Buy'
            )
            sell_plots = mpf.make_addplot(
                sell_prices, type='scatter', markersize=100, marker='v', color='red', label='Sell'
            )
            trade_plots = [buy_plots, sell_plots]

            # --- 3. Prepare Net Worth for a separate subplot ---
            net_worth_plot = mpf.make_addplot(
                history_df['net_worth'],
                panel=2,
                color='blue',
                type='line',
                ylabel='Net Worth ($)'
            )

            all_add_plots = trade_plots + [net_worth_plot]

            # --- 4. Plot and SAVE using mplfinance (The fix) ---
            file_name = f'{agent_name.lower().replace(" ", "_")}_evaluation_chart_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
            
            # Create the plot, capture the figure object, and suppress the warning
            fig, axlist = mpf.plot(
                episode_df,
                type='candle',
                style='yahoo',
                volume=True,
                addplot=all_add_plots,
                figratio=(16,9),
                title=f'{agent_name} Trading Performance',
                ylabel='Price',
                mav=(10, 50),
                show_nontrading=False,
                tight_layout=True,
                returnfig=True, 
                warn_too_much_data=len(episode_df) + 1 # Suppresses the warning
            )

            # Save the figure to the specified file name
            fig.savefig(file_name) 
            
            # Close the figure to free up memory (important when running multiple agents)
            plt.close(fig) 
            
            print(f"\nVisualisation saved to: {file_name}")

        else:
            # Fallback for other modes
            super().render(mode=mode)