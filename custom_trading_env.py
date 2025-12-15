import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import talib

class ContinuousTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000, window_size=20, commission=0.000, buy_threshold=0.0, sell_threshold=0.0):
        super(ContinuousTradingEnv, self).__init__()
        self.render_mode = 'rgb_array'

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.episode_trade_count = 0  # Counter for trades in current episode
        
        # --- FIX 1: Separate data for plotting/trading vs. observation features ---
        # 1. raw_data_for_plot: Retain absolute OHLCV and timestamp for trading and plotting.
        #    This data is used in reset(), step(), and render().
        self.raw_data_for_plot = self._clean_data_for_plot(df)
        
        # 2. self.df: Contains the normalized features (used by _get_observation for the agent).
        self.df = self._create_features(self.raw_data_for_plot.copy())
        
        # --- NEW: History logging structure ---
        self.history = self._get_history_template()

        # --- NEW: Reward enhancement tracking ---
        self.portfolio_returns = deque(maxlen=100)  # For Sharpe calculation
        self.episode_start_price = None  # For buy-and-hold benchmark

        # --- Observation Space ---
        # The agent observes a window_size of normalized market data (12 features: O,H,L,C,V,EMA, MACD, RSI, STO_RSI)
        # plus 2 internal state features (Current Balance, Current Holding Amount).
        obs_shape = (window_size * 12 + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # --- Action Space (Continuous for SAC/PPO) ---
        # Action is a single float between -1.0 (Strong Sell) and 1.0 (Strong Buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
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
            # Forward-fill NaNs with the first valid percentage change to avoid artificial zeros
            df_features.loc[:, col] = df_features[col].bfill().fillna(0)
            
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
            'reward': [],
            'action': [],
            'trade_type': [], # 'buy', 'sell', 'hold'
            'trade_price': [],
            'shares_held': [],
        }

    def _calculate_sharpe_reward(self, window=50):
        """Calculate Sharpe-like reward over a rolling window."""
        if len(self.portfolio_returns) < window:
            return 0.0

        # Get recent returns
        recent_returns = list(self.portfolio_returns)[-window:]

        # Calculate annualized Sharpe ratio components
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8  # Avoid division by zero

        # Sharpe ratio (assuming daily returns, multiply by sqrt(252) for annualization)
        sharpe = (mean_return / std_return) * np.sqrt(252)

        # Scale for reward (typically Sharpe > 1 is good, < 0 is bad)
        reward = np.clip(sharpe, -5, 5)  # Reasonable bounds

        return reward

    def _calculate_buy_hold_return(self):
        """Calculate return if holding from episode start."""
        if self.episode_start_price is None:
            self.episode_start_price = self.raw_data_for_plot.loc[self.current_step]['close']
            return 0.0

        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        # Buy-and-hold return from episode start
        price_change = (current_price - self.episode_start_price) / self.episode_start_price
        buy_hold_return = price_change * 100.0  # As percentage

        return buy_hold_return

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
        
        # Start at a random point after the first observation window, with longer burn-in to avoid NaN artifacts
        burn_in_buffer = 50  # Additional rows to skip beyond window_size
        min_start = self.window_size + burn_in_buffer
        max_start = len(self.df) - 100
        if max_start <= min_start:
            max_start = min_start + 1
        self.current_step = self.np_random.integers(min_start, max_start)

        # FIX 2: Use raw data for correct price calculation
        initial_price = self.raw_data_for_plot.loc[self.current_step]['close']
        self.max_shares = self.initial_balance / initial_price

        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance

        # New: Reset and initialize history for the new episode
        self.history = self._get_history_template()
        self.portfolio_returns.clear()  # Reset for new episode
        self.episode_start_price = self.raw_data_for_plot.loc[self.current_step]['close']  # Set for buy-and-hold benchmark

        # FIX 3: Correctly log the datetime value
        self.history['date'].append(self.raw_data_for_plot.loc[self.current_step, 'timestamp'])
        self.history['net_worth'].append(self.net_worth)
        self.history['reward'].append(0.0)  # Initial reward is 0
        self.history['action'].append(0.0)
        self.history['trade_type'].append('hold')
        self.history['trade_price'].append(np.nan)
        self.history['shares_held'].append(self.shares_held)

        observation = self._get_observation()
        info = {}
        return observation, info

    def _take_action(self, action):
        # Action comes in as a scalar
        # Clip just in case, though PPO usually handles bounds
        current_action = np.clip(action, -1, 1)

        current_price = self.raw_data_for_plot.loc[self.current_step]['close']

        trade_type = 'hold'
        trade_price = np.nan

        # SCENARIO: BUYING (Action > buy_threshold)
        if current_action > self.buy_threshold:
            # Calculate how much we can buy with available balance
            # We multiply by the action magnitude (e.g., 0.5 = use 50% of cash)
            total_possible_crypto = self.balance / current_price
            amount_to_buy = total_possible_crypto * current_action

            # Execute Buy
            cost = amount_to_buy * current_price
            trading_fee = cost * self.commission

            if self.balance >= (cost + trading_fee):
                self.balance -= (cost + trading_fee)
                self.shares_held += amount_to_buy
                trade_type = 'buy'
                trade_price = current_price

        # SCENARIO: SELLING (Action < -sell_threshold)
        elif current_action < -self.sell_threshold:
            # Calculate how much to sell based on current holdings
            # We look at the absolute value (e.g., -0.3 = sell 30% of holdings)
            amount_to_sell = self.shares_held * abs(current_action)

            if amount_to_sell > 0:
                # Execute Sell
                revenue = amount_to_sell * current_price
                trading_fee = revenue * self.commission

                self.balance += (revenue - trading_fee)
                self.shares_held -= amount_to_sell
                trade_type = 'sell'
                trade_price = current_price

        # Action == 0 or below threshold is a Hold, do nothing.
        return trade_type, trade_price

    def step(self, action):
        # Action is a single float in [-1, 1]. Positive is Buy, Negative is Sell.
        self.current_step += 1
        
        # FIX 4: Use raw data for correct price calculation
        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        action_magnitude = action[0] if hasattr(action, '__len__') else action

        # --- Execute Trade based on Action ---
        trade_type, trade_price_log = self._take_action(action_magnitude)

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

        # Calculate portfolio return for tracking
        portfolio_return = (self.net_worth - self.previous_net_worth) / (self.previous_net_worth + 1e-8)
        self.portfolio_returns.append(portfolio_return)

        # 1. Base Reward: Use Log Return for smoother, more stable training
        net_worth_ratio = self.net_worth / self.previous_net_worth
        # Use np.log(max(..., 1e-6)) to avoid log of zero/negative
        base_reward = np.log(max(net_worth_ratio, 1e-6))

        # 2. Sharpe-like Reward Component
        sharpe_reward = self._calculate_sharpe_reward(window=30) * 0.1  # Scale down Sharpe component

        # 3. Buy-and-Hold Benchmark Comparison
        buy_hold_return = self._calculate_buy_hold_return()
        benchmark_penalty = -0.01 * buy_hold_return  # Penalize if underperforming buy-and-hold

        # 4. Action Penalty: Strongly penalize any trade action to reduce over-trading
        TRADE_PENALTY_COEF = 0.001 # 0.1% penalty for any action magnitude
        action_penalty = -TRADE_PENALTY_COEF * abs(action_magnitude)

        # 5. Bankruptcy Check: Terminate episode if net worth falls below 30% of initial balance
        terminated = False
        truncated = False
        if self.net_worth < (self.initial_balance * 0.3):
            terminated = True
            reward = -10  # Large penalty for bankruptcy
        else:
            # 6. Final Reward (only if not bankrupt)
            inactivity_penalty = -0.0001 if abs(action_magnitude) < 0.1 else 0
            reward = base_reward + sharpe_reward + benchmark_penalty + action_penalty + frequency_penalty + inactivity_penalty

        # Check for truncation (end of data)
        if self.current_step >= len(self.df) - 1:
            truncated = True

        self.previous_net_worth = self.net_worth

        observation = self._get_observation()

        # FIX 5: Correctly log the datetime value
        self.history['date'].append(self.raw_data_for_plot.loc[self.current_step, 'timestamp'])
        self.history['net_worth'].append(self.net_worth)
        self.history['reward'].append(reward)  # Log the calculated reward
        self.history['action'].append(float(action_magnitude))
        self.history['trade_type'].append(trade_type)
        self.history['trade_price'].append(trade_price_log)
        self.history['shares_held'].append(self.shares_held)

        info = {
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'current_price': current_price,
            'action': action_magnitude,
            'timestamp': self.raw_data_for_plot.loc[self.current_step, 'timestamp'],
            'trade_executed': trade_executed
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', agent_name='RL_Agent'):
        """Plots the agent's performance using mplfinance, saving the result to a file."""
        print(f"DEBUG: render called with mode='{mode}', agent_name='{agent_name}'")
        print(f"DEBUG: current_step={self.current_step}, history length={len(self.history['date'])}")
        if mode == 'human':
            print("DEBUG: Rendering in human mode")
            # 1. Convert history to DataFrame and set a DatetimeIndex
            history_df = pd.DataFrame(self.history)
            print(f"DEBUG: history_df shape: {history_df.shape}")
            history_df.set_index(pd.to_datetime(history_df['date']), inplace=True)

            # Slice the raw_data_for_plot for the episode duration
            start_idx = self.current_step - len(history_df) + 1
            end_idx = self.current_step + 1
            print(f"DEBUG: slicing raw_data_for_plot from {start_idx} to {end_idx}")

            episode_df_raw = self.raw_data_for_plot.iloc[start_idx:end_idx].copy()
            print(f"DEBUG: episode_df_raw shape: {episode_df_raw.shape}")
            episode_df = episode_df_raw.set_index('timestamp').drop(columns=['ema_50', 'ema_200', 'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60'], errors='ignore')
            episode_df.index.name = 'Date'
            print(f"DEBUG: episode_df shape after processing: {episode_df.shape}")

            # --- 2. Prepare Trades for mplfinance scatter plot ---
            buys = history_df[history_df['trade_type'] == 'buy']
            sells = history_df[history_df['trade_type'] == 'sell']

            buy_prices = pd.Series(index=episode_df.index, data=np.nan)
            sell_prices = pd.Series(index=episode_df.index, data=np.nan)
            buy_prices.loc[buys.index] = buys['trade_price']
            sell_prices.loc[sells.index] = sells['trade_price']

            trade_plots = []
            if not buys.empty:
                buy_plots = mpf.make_addplot(
                    buy_prices, type='scatter', markersize=100, marker='^', color='green', label='Buy'
                )
                trade_plots.append(buy_plots)
            if not sells.empty:
                sell_plots = mpf.make_addplot(
                    sell_prices, type='scatter', markersize=100, marker='v', color='red', label='Sell'
                )
                trade_plots.append(sell_plots)

            # --- 3. Prepare Reward for a separate subplot ---
            reward_plot = mpf.make_addplot(
                history_df['reward'],
                panel=2,
                color='orange',
                type='line',
                ylabel='Reward'
            )

            # --- 4. Prepare Shares Held for a separate subplot ---
            shares_plot = mpf.make_addplot(
                history_df['shares_held'],
                panel=3,
                color='purple',
                type='line',
                ylabel='Shares Held'
            )

            all_add_plots = trade_plots + [reward_plot, shares_plot]

            # --- 4. Plot and SAVE using mplfinance (The fix) ---
            file_name = f'{agent_name.lower().replace(" ", "_")}_evaluation_chart_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'

            # Create the plot, capture the figure object, and suppress the warning
            fig, axlist = mpf.plot(
                episode_df,
                type='candle',
                style='yahoo',
                volume=True,
                addplot=all_add_plots,
                figratio=(3,1),
                figsize=(25,5),
                title=f'{agent_name} Trading Performance',
                ylabel='Price',
                mav=(10, 50),
                show_nontrading=False,
                tight_layout=False,  # Disable tight_layout for more padding
                # panelratios parameter removed in newer mplfinance versions
                returnfig=True,
                warn_too_much_data=len(episode_df) + 1 # Suppresses the warning
                # pad_rect parameter removed in newer mplfinance versions
            )
            
            # Add extra padding between subplots
            for ax in axlist:
                ax.margins(x=0.02, y=0.05)  # Add margins for better spacing

            # Add legend
            fig.legend(loc='upper left')

            # Save the figure to the specified file name
            fig.savefig(file_name) 
            
            # Close the figure to free up memory (important when running multiple agents)
            plt.close(fig) 
            
            print(f"\nVisualisation saved to: {file_name}")

        elif mode == 'rgb_array':
            print("DEBUG: Rendering in rgb_array mode")
            try:
                # Same plotting logic as human mode, but return image array instead of saving
                # 1. Convert history to DataFrame and set a DatetimeIndex
                history_df = pd.DataFrame(self.history)
                #print(f"DEBUG: rgb_array history_df shape: {history_df.shape}")
                history_df.set_index(pd.to_datetime(history_df['date']), inplace=True)

                # Slice the raw_data_for_plot for the episode duration
                start_idx = self.current_step - len(history_df) + 1
                end_idx = self.current_step + 1
                #print(f"DEBUG: rgb_array slicing from {start_idx} to {end_idx}")

                episode_df_raw = self.raw_data_for_plot.iloc[start_idx:end_idx].copy()
                #print(f"DEBUG: rgb_array episode_df_raw shape: {episode_df_raw.shape}")
                episode_df = episode_df_raw.set_index('timestamp').drop(columns=['ema_50', 'ema_200', 'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60'], errors='ignore')
                episode_df.index.name = 'Date'
                #print(f"DEBUG: rgb_array episode_df shape: {episode_df.shape}")

                # --- 2. Prepare Trades for mplfinance scatter plot ---
                buys = history_df[history_df['trade_type'] == 'buy']
                sells = history_df[history_df['trade_type'] == 'sell']

                buy_prices = pd.Series(index=episode_df.index, data=np.nan)
                sell_prices = pd.Series(index=episode_df.index, data=np.nan)
                buy_prices.loc[buys.index] = buys['trade_price']
                sell_prices.loc[sells.index] = sells['trade_price']

                trade_plots = []
                if not buys.empty:
                    buy_plots = mpf.make_addplot(
                        buy_prices, type='scatter', markersize=100, marker='^', color='green', label='Buy'
                    )
                    trade_plots.append(buy_plots)
                if not sells.empty:
                    sell_plots = mpf.make_addplot(
                        sell_prices, type='scatter', markersize=100, marker='v', color='red', label='Sell'
                    )
                    trade_plots.append(sell_plots)

                # --- 3. Prepare Reward for a separate subplot ---
                reward_plot = mpf.make_addplot(
                    history_df['reward'],
                    panel=2,
                    color='orange',
                    type='line',
                    ylabel='Reward'
                )

                # --- 4. Prepare Shares Held for a separate subplot ---
                shares_plot = mpf.make_addplot(
                    history_df['shares_held'],
                    panel=3,
                    color='purple',
                    type='line',
                    ylabel='Shares Held'
                )

                all_add_plots = trade_plots + [reward_plot, shares_plot]

                # --- 4. Plot using mplfinance, get figure ---
                fig, axlist = mpf.plot(
                    episode_df,
                    type='candle',
                    style='yahoo',
                    volume=True,
                    addplot=all_add_plots,
                    figratio=(3,1),
                    figsize=(25,5),
                    title=f'{agent_name} Trading Performance',
                    ylabel='Price',
                    mav=(10, 50),
                    show_nontrading=False,
                    tight_layout=False,  # Disable tight_layout for more padding
                    # panelratios parameter removed in newer mplfinance versions
                    returnfig=True,
                    warn_too_much_data=len(episode_df) + 1,
                    # pad_rect parameter removed in newer mplfinance versions
                )
                
                # Add extra padding between subplots
                for ax in axlist:
                    ax.margins(x=0.02, y=0.05)  # Add margins for better spacing

                # Add legend
                fig.legend(loc='upper left')

                # Convert figure to RGB array
                fig.canvas.draw()
                width, height = fig.get_size_inches() * fig.dpi
                width = int(width)
                height = int(height)
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

                # Close the figure to free up memory
                plt.close(fig)

                #print(f"DEBUG: Returning image array of shape {img_array.shape}")
                return img_array
            except Exception as e:
                print(f"DEBUG: Error in rgb_array render: {e}")
                import traceback
                traceback.print_exc()
                return None

        else:
            # Fallback for other modes
            print(f"DEBUG: Unsupported render mode '{mode}', calling super().render() which likely returns None")
            result = super().render(mode=mode)
            print(f"DEBUG: super().render() returned: {result}")
            return result