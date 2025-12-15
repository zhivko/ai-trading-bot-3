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
from datetime import datetime, timedelta

class ImprovedTradingEnv(gym.Env):
    """
    Improved Trading Environment with Fixed Reward Structure
    
    Key Improvements:
    1. Symmetric fee penalties (0.15% consistent rate)
    2. Symmetric hold duration penalties (24h for crypto)
    3. Improved action change penalties (only unnecessary changes)
    4. Reward component balancing (base reward dominance)
    5. Position size penalties (80% max exposure)
    6. Market condition awareness (volatility-based adjustments)
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000, window_size=20,
                  trading_fee_rate=0.0015, max_exposure=0.8, optimal_hold_duration=24,
                  max_hold_steps=20):  # New param: max_hold_steps
        super(ImprovedTradingEnv, self).__init__()
        self.render_mode = 'rgb_array'

        # Core configuration
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.trading_fee_rate = trading_fee_rate  # 0.15% = 0.0015
        self.max_exposure = max_exposure  # 80% maximum exposure
        self.optimal_hold_duration = optimal_hold_duration  # 24 hours for crypto
        self.max_hold_steps = max_hold_steps  # New: maximum hold steps (e.g., 10 hours)
        
        # Initialize data
        self.raw_data_for_plot = self._clean_data_for_plot(df)
        self.df = self._create_features(self.raw_data_for_plot.copy())
        
        # Initialize history tracking
        self.history = self._get_history_template()
        
        # Episode tracking
        self.episode_trade_count = 0
        self.position_start_step = None
        self.position_type = None  # 'long', 'short', or None
        self.trade_statistics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_fees_paid': 0.0,
            'total_holding_penalties': 0.0,
            'total_action_penalties': 0.0,
            'total_position_penalties': 0.0
        }

        # Market condition tracking
        self.volatility_window = 20
        self.recent_volatility = deque(maxlen=self.volatility_window)
        self.market_regime = 'normal'  # 'low', 'normal', 'high'
        
        # Portfolio returns for Sharpe calculation
        self.portfolio_returns = deque(maxlen=100)
        self.episode_start_price = None

        # --- Observation Space ---
        # Window of normalized market data + account state
        obs_shape = (window_size * 12 + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # --- Multi-dimensional Action Space ---
        # Dimension 1: Position target (-1 to 1, where -1=max short, 0=flat, 1=max long)
        # Dimension 2: Position size intensity (0 to 1, scaling factor for position size)
        # Dimension 3: Hold duration (0 to max_hold_steps, mapped to 0-max_hold_steps for locking)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 10.0]),  # Changed low[2] to 10.0 for minimum hold duration
            high=np.array([1.0, 1.0, self.max_hold_steps]),
            shape=(3,),
            dtype=np.float32
        )
        
        self.reset()

    def _clean_data_for_plot(self, df):
        """Prepares raw data by cleaning and setting timestamp."""
        df_cleaned = df.copy()
        df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'])
        df_cleaned = df_cleaned.set_index('timestamp')
        df_cleaned = df_cleaned.dropna()
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

        # Calculate volatility for market condition awareness
        df_features['returns'] = df_features['close'].pct_change()
        df_features['volatility'] = df_features['returns'].rolling(window=20).std()

        # Normalize Price Data and EMA
        price_cols = ['open', 'high', 'low', 'close', 'ema_50', 'ema_200']
        for col in price_cols:
            df_features.loc[:, col] = df_features[col].pct_change()
            df_features.loc[:, col] = df_features[col].bfill().fillna(0)
            
        # Normalize volume
        df_features.loc[:, 'volume'] = df_features['volume'] / df_features['volume'].rolling(self.window_size).mean()
        
        # Normalize MACD indicators
        df_features.loc[:, 'macd'] = df_features['macd'] / df_features['macd'].rolling(self.window_size).std().fillna(1)
        df_features.loc[:, 'macd_signal'] = df_features['macd_signal'] / df_features['macd_signal'].rolling(self.window_size).std().fillna(1)
        
        # RSI and Stochastic RSI normalization
        df_features.loc[:, 'rsi'] = df_features['rsi'] / 100.0
        df_features.loc[:, 'sto_rsi_3_14'] = df_features['sto_rsi_3_14'] / 100.0
        df_features.loc[:, 'sto_rsi_10_60'] = df_features['sto_rsi_10_60'] / 100.0
        
        # Volatility normalization
        df_features.loc[:, 'volatility'] = df_features['volatility'] / df_features['volatility'].rolling(50).mean().fillna(0.01)
        
        df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)
        if 'timestamp' in df_features.columns:
            df_features = df_features.drop(columns=['timestamp'])
        return df_features

    def _get_history_template(self):
        """Creates the dictionary structure for logging episode history."""
        return {
            'date': [],
            'net_worth': [],
            'reward': [],
            'reward_components': {},
            'action': [],
            'action_components': [],
            'trade_type': [],
            'trade_price': [],
            'shares_held': [],
            'position_size': [],
            'fees_paid': [],
            'penalties_applied': {}
        }

    def _calculate_market_regime(self):
        """Calculate current market regime based on volatility."""
        if len(self.recent_volatility) < 10:
            return 'normal'
        
        current_vol = np.mean(list(self.recent_volatility)[-10:])
        vol_history = list(self.recent_volatility)
        
        if len(vol_history) >= 20:
            vol_percentile = np.percentile(vol_history, 75)
            if current_vol > vol_percentile * 1.5:
                return 'high'
            elif current_vol < np.percentile(vol_history, 25) * 0.5:
                return 'low'
        
        return 'normal'

    def _calculate_volatility_adjustment(self):
        """Calculate volatility-based adjustment for penalties."""
        regime_multipliers = {
            'low': 0.8,    # Reduce penalties in low volatility
            'normal': 1.0,  # Normal penalties
            'high': 1.3     # Increase penalties in high volatility
        }
        return regime_multipliers.get(self.market_regime, 1.0)

    def _get_observation(self):
        """Get normalized observation for the agent."""
        # Extract features for the current time step window
        window_data = self.df.loc[self.current_step - self.window_size + 1:self.current_step,
                                  ['open', 'high', 'low', 'close', 'volume', 'ema_50', 'ema_200', 
                                   'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60']].values.flatten()
        
        # Append internal state (Normalized Balance and Holding)
        state_features = np.array([
            self.balance / self.initial_balance,
            self.shares_held / self.max_shares
        ])
        
        return np.concatenate((window_data, state_features))

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        
        # Reset financial state
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        
        # Reset episode tracking
        self.episode_trade_count = 0
        self.position_start_step = None
        self.position_type = None
        
        # Reset lock tracking
        self.current_lock_duration = 0
        self.last_position_target = 0.0
        self.last_size_intensity = 0.0
        
        # Reset trade statistics
        self.trade_statistics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_fees_paid': 0.0,
            'total_holding_penalties': 0.0,
            'total_action_penalties': 0.0,
            'total_position_penalties': 0.0,
            'total_lock_bonuses': 0.0
        }
        
        # Reset market tracking
        self.recent_volatility.clear()
        self.market_regime = 'normal'
        
        # Random start position
        burn_in_buffer = 50
        min_start = self.window_size + burn_in_buffer
        max_start = len(self.df) - 100
        if max_start <= min_start:
            max_start = min_start + 1
        self.current_step = self.np_random.integers(min_start, max_start)
        
        # Initialize max shares based on current price
        initial_price = self.raw_data_for_plot.loc[self.current_step]['close']
        self.max_shares = (self.initial_balance * self.max_exposure) / initial_price
        
        # Reset history
        self.history = self._get_history_template()
        self.portfolio_returns.clear()
        self.episode_start_price = initial_price
        
        # Log initial state
        self.history['date'].append(self.raw_data_for_plot.loc[self.current_step, 'timestamp'])
        self.history['net_worth'].append(self.net_worth)
        self.history['reward'].append(0.0)
        self.history['action'].append([0.0, 0.0, 0.0])
        self.history['action_components'].append([0.0, 0.0, 0.0])
        self.history['trade_type'].append('hold')
        self.history['trade_price'].append(np.nan)
        self.history['shares_held'].append(self.shares_held)
        self.history['position_size'].append(0.0)
        self.history['fees_paid'].append(0.0)
        self.history['penalties_applied'] = {
            'fee_penalty': 0.0,
            'holding_penalty': 0.0,
            'action_change_penalty': 0.0,
            'position_size_penalty': 0.0,
            'duration_penalty': 0.0,
            'lock_bonus': 0.0
        }
        
        observation = self._get_observation()
        info = self._get_info_dict()
        return observation, info

    def _get_info_dict(self):
        """Get comprehensive info dictionary with debugging information."""
        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        
        # Calculate position metrics
        position_value = abs(self.shares_held * current_price)
        exposure_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
        
        # Calculate time in position
        time_in_position = 0
        if self.position_start_step is not None:
            time_in_position = self.current_step - self.position_start_step
        
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position_value': position_value,
            'exposure_ratio': exposure_ratio,
            'current_price': current_price,
            'market_regime': self.market_regime,
            'volatility_adjustment': self._calculate_volatility_adjustment(),
            'episode_trade_count': self.episode_trade_count,
            'time_in_position': time_in_position,
            'trade_statistics': self.trade_statistics.copy(),
            'timestamp': str(self.raw_data_for_plot.loc[self.current_step, 'timestamp']),
            'reward_components': {},  # Will be populated in step()
            'action_components': [0.0, 0.0, 0.0],  # Will be populated in step()
            'trade_executed': False
        }

    def _execute_trade(self, action):
        """Execute trade based on multi-dimensional action."""
        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        
        # Extract action components
        position_target = np.clip(action[0], -1.0, 1.0)  # -1 to 1
        size_intensity = np.clip(action[1], 0.0, 1.0)    # 0 to 1
        duration_preference = np.clip(action[2], 10.0, self.max_hold_steps)  # 10 to max_hold_steps (duration)
        
        # Calculate target position size
        max_position_value = self.net_worth * self.max_exposure
        target_position_value = max_position_value * size_intensity * abs(position_target)
        
        # Calculate current position value
        current_position_value = abs(self.shares_held * current_price)
        
        # Determine if we need to trade
        position_diff = target_position_value - current_position_value
        
        # Check minimum trade threshold
        min_trade_value = self.initial_balance * 0.001  # 0.1% of initial balance
        
        trade_executed = False
        fees_paid = 0.0
        
        if abs(position_diff) > min_trade_value:
            if position_diff > 0:  # Need to buy
                # Calculate shares to buy
                shares_to_buy = position_diff / current_price
                trade_value = shares_to_buy * current_price
                
                # Check if we have enough balance
                max_affordable = self.balance * 0.95  # Leave some buffer
                if trade_value <= max_affordable:
                    # Execute buy
                    self.balance -= trade_value
                    self.shares_held += shares_to_buy
                    
                    # Calculate and track fees
                    fees = trade_value * self.trading_fee_rate
                    self.balance -= fees
                    fees_paid = fees
                    
                    trade_executed = True
                    
            elif position_diff < 0:  # Need to sell
                # Calculate shares to sell
                shares_to_sell = min(abs(position_diff) / current_price, abs(self.shares_held))
                
                if shares_to_sell > 0:
                    # Execute sell
                    self.balance += shares_to_sell * current_price
                    self.shares_held -= shares_to_sell * np.sign(self.shares_held)
                    
                    # Calculate and track fees
                    trade_value = shares_to_sell * current_price
                    fees = trade_value * self.trading_fee_rate
                    self.balance -= fees
                    fees_paid = fees
                    
                    trade_executed = True
        
        # Update position tracking
        if trade_executed:
            self.episode_trade_count += 1
            self.trade_statistics['total_trades'] += 1
            self.trade_statistics['total_fees_paid'] += fees_paid
            
            # Update position type and start step
            if self.shares_held > 0:
                self.position_type = 'long'
            elif self.shares_held < 0:
                self.position_type = 'short'
            else:
                self.position_type = None
            
            self.position_start_step = self.current_step
        
        return trade_executed, fees_paid, [position_target, size_intensity, duration_preference]

    def _calculate_reward_components(self, trade_executed, fees_paid, action_components):
        """Calculate all reward components with improved structure."""

        # Calculate hold preference (normalized duration)
        hold_preference = action_components[2] / self.max_hold_steps

        # 1. Base Reward: Net worth change percentage
        base_reward = 0.0
        if self.previous_net_worth > 0:
            base_reward = ((self.net_worth - self.previous_net_worth) / self.previous_net_worth) * 100.0
        
        # 2. Fee Penalty: Symmetric application of actual trading costs
        fee_penalty = -abs(fees_paid / self.net_worth * 100.0) if fees_paid > 0 else 0.0
        fee_penalty *= self.trading_fee_rate * 100  # Scale to match reward magnitude
        
        # 3. Action Change Penalty: Only penalize unnecessary changes
        action_change_penalty = 0.0
        if hasattr(self, 'previous_action'):
            prev_pos = self.previous_action[0] if len(self.previous_action) > 0 else 0.0
            curr_pos = action_components[0]
            action_delta = abs(curr_pos - prev_pos)
            
            # Only penalize noise changes (0.1-0.3) and excessive churning (>0.8)
            if 0.1 <= action_delta <= 0.3:
                action_change_penalty = -action_delta * 0.5  # Moderate penalty for noise
            elif action_delta > 0.8:
                action_change_penalty = -action_delta * 1.0  # Higher penalty for excessive churning
            # Small changes (<0.1) are free (encourages fine-tuning)
        
        # 4. Symmetric Hold Duration Penalty: Applied equally to long and short
        holding_penalty = 0.0
        duration_penalty = 0.0
        if self.position_start_step is not None and self.position_type is not None:
            time_in_position = self.current_step - self.position_start_step
            
            # Penalize holding beyond optimal duration
            if time_in_position > self.optimal_hold_duration:
                excess_time = abs(time_in_position - self.optimal_hold_duration)
                duration_penalty = -(excess_time / self.optimal_hold_duration) * 0.1
                
            # Symmetric holding penalty for any position
            position_value = abs(self.shares_held * self.raw_data_for_plot.loc[self.current_step]['close'])
            position_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
            holding_penalty = -position_ratio * time_in_position * 0.001
        
        # 5. Position Size Penalty: Penalize excessive leverage
        position_size_penalty = 0.0
        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        position_value = abs(self.shares_held * current_price)
        exposure_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
        
        if exposure_ratio > self.max_exposure:
            excess_exposure = exposure_ratio - self.max_exposure
            position_size_penalty = -excess_exposure * 10.0  # Strong penalty for over-exposure
        
        # 6. Market Condition Adjustment
        volatility_adjustment = self._calculate_volatility_adjustment()
        
        # Apply market condition adjustments to penalties
        fee_penalty *= volatility_adjustment
        action_change_penalty *= volatility_adjustment
        holding_penalty *= volatility_adjustment
        duration_penalty *= volatility_adjustment
        position_size_penalty *= volatility_adjustment
        
        # 7. Reward Component Balancing: Ensure base reward dominates
        total_penalties = abs(fee_penalty) + abs(action_change_penalty) + abs(holding_penalty) + abs(duration_penalty) + abs(position_size_penalty)
        
        if abs(base_reward) > 1e-6 and total_penalties > abs(base_reward) * 0.5:
            # Scale down penalties if they exceed 50% of base reward
            max_penalties = abs(base_reward) * 0.5
            if total_penalties > max_penalties:
                reduction_factor = max_penalties / total_penalties
                fee_penalty *= reduction_factor
                action_change_penalty *= reduction_factor
                holding_penalty *= reduction_factor
                duration_penalty *= reduction_factor
                position_size_penalty *= reduction_factor

        # Hold preference tweaks
        hold_bonus = 0.0
        hold_penalty = 0.0
        if self.market_regime == 'low':
            hold_bonus = 0.002 * hold_preference
        if self.market_regime == 'high' and hold_preference < 0.5:
            hold_penalty = -0.001

        # Compile reward components
        reward_components = {
            'base': base_reward,
            'fee_penalty': fee_penalty,
            'action_change_penalty': action_change_penalty,
            'holding_penalty': holding_penalty,
            'duration_penalty': duration_penalty,
            'position_size_penalty': position_size_penalty,
            'hold_bonus': hold_bonus,
            'hold_penalty': hold_penalty
        }
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Update statistics
        self.trade_statistics['total_holding_penalties'] += abs(holding_penalty) + abs(duration_penalty)
        self.trade_statistics['total_action_penalties'] += abs(action_change_penalty)
        self.trade_statistics['total_position_penalties'] += abs(position_size_penalty)
        
        return total_reward, reward_components

    def step(self, action):
        """Execute one time step in the environment with duration-based locking."""
        self.current_step += 1
        
        # Check for episode end
        if self.current_step >= len(self.df) - 1:
            observation = self._get_observation()
            return observation, 0.0, False, True, self._get_info_dict()
        
        # Update market regime
        current_volatility = self.df.loc[self.current_step]['volatility']
        self.recent_volatility.append(current_volatility)
        self.market_regime = self._calculate_market_regime()
        
        # Store previous state
        self.previous_net_worth = self.net_worth
        if hasattr(self, 'previous_action'):
            pass  # Will be used for action change penalty
        else:
            self.previous_action = [0.0, 0.0, 0.0]
        
        # LOCK LOGIC: Check if we're in a lock period
        trade_executed = False
        fees_paid = 0.0
        action_components = [0.0, 0.0, 0.0]
        reward_bonus = 0.0
        
        if self.current_lock_duration > 0:
            # We're locked - ignore new action, hold current position
            self.current_lock_duration -= 1
            action_components = [self.last_position_target, self.last_size_intensity, 0.0]
            
            # Optional: Small bonus for holding if position is profitable
            if self.net_worth > self.previous_net_worth:
                reward_bonus = 0.0005  # Small bonus for profitable hold
        else:
            # Not locked - process new action
            trade_executed, fees_paid, action_components = self._execute_trade(action)
            
            # Set new lock if duration > 0
            if len(action_components) >= 3 and action_components[2] > 0:
                hold_steps = int(action_components[2])
                if hold_steps > 0:
                    self.current_lock_duration = hold_steps - 1  # Current step executes the action
                    self.last_position_target = action_components[0]
                    self.last_size_intensity = action_components[1]
        
        # Update net worth (always, based on current position)
        current_price = self.raw_data_for_plot.loc[self.current_step]['close']
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate portfolio return
        portfolio_return = (self.net_worth - self.previous_net_worth) / (self.previous_net_worth + 1e-8)
        self.portfolio_returns.append(portfolio_return)
        
        # Calculate reward components
        reward, reward_components = self._calculate_reward_components(trade_executed, fees_paid, action_components)
        
        # Add lock bonus if applicable
        reward += reward_bonus
        if reward_bonus > 0:
            reward_components['lock_bonus'] = reward_bonus
            self.trade_statistics['total_lock_bonuses'] += reward_bonus
        
        # Update action history
        self.previous_action = action_components
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Bankruptcy check
        if self.net_worth < self.initial_balance * 0.3:
            terminated = True
            reward = -10.0  # Large penalty for bankruptcy
        
        # Check for end of data
        if self.current_step >= len(self.df) - 1:
            truncated = True
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = self._get_info_dict()
        info['reward_components'] = reward_components
        info['action_components'] = action_components
        info['trade_executed'] = trade_executed
        info['fees_paid'] = fees_paid
        info['lock_remaining'] = self.current_lock_duration
        
        # Update history
        self.history['date'].append(self.raw_data_for_plot.loc[self.current_step, 'timestamp'])
        self.history['net_worth'].append(self.net_worth)
        self.history['reward'].append(reward)
        self.history['reward_components'] = reward_components
        self.history['action'].append(action)
        self.history['action_components'].append(action_components)
        
        trade_type = 'hold'
        if trade_executed:
            if action_components[0] > 0:
                trade_type = 'buy'
            elif action_components[0] < 0:
                trade_type = 'sell'
        
        self.history['trade_type'].append(trade_type)
        self.history['trade_price'].append(current_price if trade_executed else np.nan)
        self.history['shares_held'].append(self.shares_held)
        self.history['position_size'].append(abs(self.shares_held * current_price))
        self.history['fees_paid'].append(fees_paid)
        self.history['penalties_applied'] = reward_components
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human', agent_name='Improved_RL_Agent'):
        """Render the trading performance with enhanced visualization."""
        if mode == 'human':
            print(f"Rendering {agent_name} performance...")
            
            # Convert history to DataFrame (filter out problematic dict columns)
            history_data = {}
            for key, value in self.history.items():
                if isinstance(value, dict):
                    # Skip dict columns as they can't be directly converted to DataFrame
                    continue
                history_data[key] = value
            
            history_df = pd.DataFrame(history_data)
            history_df.set_index(pd.to_datetime(history_df['date']), inplace=True)
            
            # Prepare data for plotting
            start_idx = self.current_step - len(history_df) + 1
            end_idx = self.current_step + 1
            episode_df_raw = self.raw_data_for_plot.iloc[start_idx:end_idx].copy()
            episode_df = episode_df_raw.set_index('timestamp').drop(
                columns=['ema_50', 'ema_200', 'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60'], 
                errors='ignore'
            )
            episode_df.index.name = 'Date'
            
            # Prepare trade plots
            buys = history_df[history_df['trade_type'] == 'buy']
            sells = history_df[history_df['trade_type'] == 'sell']
            
            trade_plots = []
            if not buys.empty:
                buy_prices = pd.Series(index=episode_df.index, data=np.nan)
                buy_prices.loc[buys.index] = buys['trade_price']
                trade_plots.append(mpf.make_addplot(
                    buy_prices, type='scatter', markersize=100, marker='^', 
                    color='green', label='Buy'
                ))
            
            if not sells.empty:
                sell_prices = pd.Series(index=episode_df.index, data=np.nan)
                sell_prices.loc[sells.index] = sells['trade_price']
                trade_plots.append(mpf.make_addplot(
                    sell_prices, type='scatter', markersize=100, marker='v', 
                    color='red', label='Sell'
                ))
            
            # Prepare subplots
            reward_plot = mpf.make_addplot(
                history_df['reward'], panel=2, color='orange', type='line', ylabel='Reward'
            )
            
            position_size_plot = mpf.make_addplot(
                history_df['position_size'], panel=3, color='purple', type='line', ylabel='Position Size'
            )
            
            action_plot = mpf.make_addplot(
                [a[0] for a in history_df['action_components']], panel=4, 
                color='blue', type='line', ylabel='Position Target'
            )
            
            net_worth_plot = mpf.make_addplot(
                history_df['net_worth'], panel=5, color='red', type='line', ylabel='Net Worth'
            )
            
            all_add_plots = trade_plots + [reward_plot, position_size_plot, action_plot, net_worth_plot]
            
            # Create plot
            fig, axlist = mpf.plot(
                episode_df, type='candle', style='yahoo', volume=True,
                addplot=all_add_plots, figratio=(2.5, 1), figsize=(32, 14),
                title=f'{agent_name} Trading Performance (Improved Environment)',
                ylabel='Price', mav=(10, 50), show_nontrading=False,
                tight_layout=False, returnfig=True
            )
            
            # Enhance visualization
            for ax in axlist:
                ax.margins(x=0.02, y=0.05)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.tick_params(axis='both', which='minor', labelsize=5)
            
            fig.legend(loc='upper left')
            
            # Save plot
            filename = f'{agent_name.lower().replace(" ", "_")}_improved_evaluation_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
            fig.savefig(filename)
            plt.close(fig)
            
            print(f"Enhanced visualization saved to: {filename}")
            
            # Print trading statistics
            print(f"\n=== Trading Statistics ===")
            print(f"Total Trades: {self.trade_statistics['total_trades']}")
            print(f"Total Fees Paid: ${self.trade_statistics['total_fees_paid']:.2f}")
            print(f"Total Holding Penalties: {self.trade_statistics['total_holding_penalties']:.4f}")
            print(f"Total Action Penalties: {self.trade_statistics['total_action_penalties']:.4f}")
            print(f"Total Position Penalties: {self.trade_statistics['total_position_penalties']:.4f}")
            print(f"Final Net Worth: ${self.net_worth:.2f}")
            print(f"Market Regime: {self.market_regime}")
            
        elif mode == 'rgb_array':
            # Return RGB array for rendering
            # Do the same plotting as human mode but return image array
            print(f"Rendering {agent_name} performance...")

            # Convert history to DataFrame (filter out problematic dict columns)
            history_data = {}
            for key, value in self.history.items():
                if isinstance(value, dict):
                    # Skip dict columns as they can't be directly converted to DataFrame
                    continue
                history_data[key] = value

            history_df = pd.DataFrame(history_data)
            history_df.set_index(pd.to_datetime(history_df['date']), inplace=True)

            # Prepare data for plotting
            start_idx = self.current_step - len(history_df) + 1
            end_idx = self.current_step + 1
            episode_df_raw = self.raw_data_for_plot.iloc[start_idx:end_idx].copy()
            episode_df = episode_df_raw.set_index('timestamp').drop(
                columns=['ema_50', 'ema_200', 'macd', 'macd_signal', 'rsi', 'sto_rsi_3_14', 'sto_rsi_10_60'],
                errors='ignore'
            )
            episode_df.index.name = 'Date'

            # Prepare trade plots
            buys = history_df[history_df['trade_type'] == 'buy']
            sells = history_df[history_df['trade_type'] == 'sell']

            trade_plots = []
            if not buys.empty:
                buy_prices = pd.Series(index=episode_df.index, data=np.nan)
                buy_prices.loc[buys.index] = buys['trade_price']
                trade_plots.append(mpf.make_addplot(
                    buy_prices, type='scatter', markersize=100, marker='^',
                    color='green', label='Buy'
                ))

            if not sells.empty:
                sell_prices = pd.Series(index=episode_df.index, data=np.nan)
                sell_prices.loc[sells.index] = sells['trade_price']
                trade_plots.append(mpf.make_addplot(
                    sell_prices, type='scatter', markersize=100, marker='v',
                    color='red', label='Sell'
                ))

            # Prepare subplots
            reward_plot = mpf.make_addplot(
                history_df['reward'], panel=2, color='orange', type='line', ylabel='Reward'
            )

            position_size_plot = mpf.make_addplot(
                history_df['position_size'], panel=3, color='purple', type='line', ylabel='Position Size'
            )

            action_plot = mpf.make_addplot(
                [a[0] for a in history_df['action_components']], panel=4,
                color='blue', type='line', ylabel='Position Target'
            )

            net_worth_plot = mpf.make_addplot(
                history_df['net_worth'], panel=5, color='red', type='line', ylabel='Net Worth'
            )

            all_add_plots = trade_plots + [reward_plot, position_size_plot, action_plot, net_worth_plot]

            # Create plot
            fig, axlist = mpf.plot(
                episode_df, type='candle', style='yahoo', volume=True,
                addplot=all_add_plots, figratio=(2.5, 1), figsize=(32, 14),
                title=f'{agent_name} Trading Performance (Improved Environment)',
                ylabel='Price', mav=(10, 50), show_nontrading=False,
                tight_layout=False, returnfig=True
            )

            # Enhance visualization
            for ax in axlist:
                ax.margins(x=0.02, y=0.05)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.tick_params(axis='both', which='minor', labelsize=5)

            fig.legend(loc='upper left')

            # Convert to RGB array
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # Convert RGBA to RGB
            plt.close(fig)

            return img
        
        return None

    def close(self):
        """Clean up resources."""
        plt.close('all')

