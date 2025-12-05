Create chinesetrader.py that will implement below code and start flask server and visualize ohlc data and signals when to enter and when to exit trades, similarly like we do it in backtest.py
use existing requirements.txt file and use existing python environment that is in .venv directory. Dont forget you are running in windows.

# macd_money_map_btc_1h_2017_2025.py
# Full MACD Money Map Strategy Backtest + Interactive Visualization
# Based exactly on the 3-System method from the viral video

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings("ignore")

# ====================== 1. LOAD DATA ======================
df = pd.read_csv('btc_1h_with_ema50.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume', 'ema_50']]
df = df.sort_index()

# ====================== 2. MACD CALCULATION (TradingView style) ======================
def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

df['macd'], df['signal'], df['hist'] = macd(df['close'])

# Resample to 4H and Daily for multi-timeframe
df_4h = df.resample('4H').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()
df_4h['macd_4h'], df_4h['signal_4h'], _ = macd(df_4h['close'])

df_daily = df.resample('1D').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()
df_daily['macd_daily'], _, _ = macd(df_daily['close'])

# Merge higher TF bias
df = df.join(df_daily['macd_daily'].rename('macd_daily'), how='left').ffill()
df = df.join(df_4h['macd_4h'].rename('macd_4h'), how='left').ffill()

# ====================== 3. SIGNAL DETECTION FUNCTIONS ======================
def detect_divergence(price, macd_line, window=10, order=5):
    # Find local highs and lows
    high_idx = argrelextrema(price.values, np.greater, order=order)[0]
    low_idx = argrelextrema(price.values, np.less, order=order)[0]
    
    bearish_div = []
    bullish_div = []
    
    # Bearish divergence
    if len(high_idx) >= 2:
        for i in range(len(high_idx)-1):
            p1, p2 = high_idx[i], high_idx[i+1]
            if price.iloc[p2] > price.iloc[p1] and macd_line.iloc[p2] < macd_line.iloc[p1]:
                bearish_div.append(df.index[p2])
    
    # Bullish divergence
    if len(low_idx) >= 2:
        for i in range(len(low_idx)-1):
            p1, p2 = low_idx[i], low_idx[i+1]
            if price.iloc[p2] < price.iloc[p1] and macd_line.iloc[p2] > macd_line.iloc[p1]:
                bullish_div.append(df.index[p2])
    
    return bearish_div, bullish_div

# Histogram patterns
df['hist_prev'] = df['hist'].shift(1)
df['hist_flip_up'] = (df['hist_prev'] < 0) & (df['hist'] > 0)
df['hist_flip_down'] = (df['hist_prev'] > 0) & (df['hist'] < 0)
df['hist_shrinking'] = abs(df['hist']) < abs(df['hist_prev'])

# Crossovers
df['bull_cross'] = (df['macd'].shift(1) < df['signal'].shift(1)) & (df['macd'] > df['signal'])
df['bear_cross'] = (df['macd'].shift(1) > df['signal'].shift(1)) & (df['macd'] < df['signal'])

# Distance from zero (scaled to BTC price - use percentile)
distance_threshold = df['macd'].abs().rolling(500).quantile(0.75).fillna(100)

# ====================== 4. BACKTEST ENGINE ======================
trades = []
position = None

for i in range(100, len(df)-20):  # avoid edge
    row = df.iloc[i]
    prev = df.iloc[i-1]
    idx = df.index[i]
    
    # Multi-timeframe alignment
    daily_bias_long = row['macd_daily'] > 0
    daily_bias_short = row['macd_daily'] < 0
    tf_aligned_long = daily_bias_long and row['macd_4h'] > 0
    tf_aligned_short = daily_bias_short and row['macd_4h'] < 0

    # Price at key level? (simple: near EMA50 or recent swing)
    near_support = abs(row['close'] - row['ema_50']) / row['close'] < 0.02
    near_resistance = near_support  # simplified

    # === SYSTEM 1: TREND CONTINUATION ===
    valid_distance = abs(row['macd']) > distance_threshold.iloc[i]
    
    # Wait 2 bars after crossover
    if i > 2:
        bull_cross_2ago = df.iloc[i-2]['bull_cross'] and valid_distance and tf_aligned_long
        bear_cross_2ago = df.iloc[i-2]['bear_cross'] and valid_distance and tf_aligned_short
        
        if bull_cross_2ago and position is None:
            sl = row['low'] - 50  # conservative
            tp = row['close'] + 2 * (row['close'] - sl)
            position = {'entry': row['close'], 'sl': sl, 'tp': tp, 'type': 'long', 'time': idx, 'system': 'Trend'}
            
        elif bear_cross_2ago and position is None:
            sl = row['high'] + 50
            tp = row['close'] - 2 * (sl - row['close'])
            position = {'entry': row['close'], 'sl': sl, 'tp': tp, 'type': 'short', 'time': idx, 'system': 'Trend'}

    # === SYSTEM 2: REVERSAL (with histogram confirm) ===
    if i > 50:
        lookback = df.iloc[i-50:i]
        bear_div, bull_div = detect_divergence(lookback['high'], lookback['macd'])
        if idx in bear_div and row['hist_flip_down'] and tf_aligned_short and position is None:
            sl = row['high'] + 100
            tp = row['close'] - 2 * (sl - row['close'])
            position = {'entry': row['close'], 'sl': sl, 'tp': tp, 'type': 'short', 'time': idx, 'system': 'Reversal'}
            
        if idx in bull_div and row['hist_flip_up'] and tf_aligned_long and position is None:
            sl = row['low'] - 100
            tp = row['close'] + 2 * (row['close'] - sl)
            position = {'entry': row['close'], 'sl': sl, 'tp': tp, 'type': 'long', 'time': idx, 'system': 'Reversal'}

    # === EXIT LOGIC ===
    if position is not None:
        if position['type'] == 'long':
            if row['low'] <= position['sl']:
                profit = position['sl'] - position['entry']
                trades.append({**position, 'exit': idx, 'profit': profit, 'pct': profit/position['entry']})
                position = None
            elif row['high'] >= position['tp']:
                profit = position['tp'] - position['entry']
                trades.append({**position, 'exit': idx, 'profit': profit, 'pct': profit/position['entry']})
                position = None
        else:
            if row['high'] >= position['sl']:
                profit = position['entry'] - position['sl']
                trades.append({**position, 'exit': idx, 'profit': profit, 'pct': profit/position['entry']})
                position = None
            elif row['low'] <= position['tp']:
                profit = position['entry'] - position['tp']
                trades.append({**position, 'exit': idx, 'profit': profit, 'pct': profit/position['entry']})
                position = None

# ====================== 5. RESULTS ======================
trades_df = pd.DataFrame(trades)
if len(trades_df) > 0:
    win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
    total_return = trades_df['pct'].sum()
    profit_factor = trades_df[trades_df['profit']>0]['profit'].sum() / abs(trades_df[trades_df['profit']<0]['profit'].sum() or 1)
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total Return: {total_return:.1%}")
    print(f"Profit Factor: {profit_factor:.2f}")
else:
    print("No trades triggered")

# ====================== 6. PLOTLY VISUALIZATION ======================
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    subplot_titles=('BTC/USDT 1H + MACD Money Map Signals', 'MACD + Signal + Histogram', 'Daily MACD Bias', 'Equity Curve'),
    row_heights=[0.5, 0.2, 0.15, 0.15]
)

# Candles
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='BTC'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50', line=dict(color='orange')), row=1, col=1)

# Mark entries
if len(trades_df) > 0:
    longs = trades_df[trades_df['type']=='long']
    shorts = trades_df[trades_df['type']=='short']
    fig.add_trace(go.Scatter(x=longs['time'], y=longs['entry']*0.995, mode='markers', marker=dict(symbol='triangle-up', size=16, color='lime'), name='Long Entry'), row=1, col=1)
    fig.add_trace(go.Scatter(x=shorts['time'], y=shorts['entry']*1.005, mode='markers', marker=dict(symbol='triangle-down', size=16, color='red'), name='Short Entry'), row=1, col=1)

# MACD Panel
fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['signal'], name='Signal', line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['hist'], name='Histogram', marker_color=np.where(df['hist']>0, 'green', 'red')), row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=1)
fig.add_hline(y=distance_threshold.mean(), line_dash="dot", line_color="yellow", row=2, col=1, annotation_text="Distance Threshold")

# Daily MACD
fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['macd_daily'], name='Daily MACD', line=dict(width=3)), row=3, col=1)
fig.add_hline(y=0, line_color="white", row=3, col=1)

# Equity
if len(trades_df) > 0:
    equity = (1 + trades_df['pct']).cumprod()
    fig.add_trace(go.Scatter(x=trades_df['time'], y=equity*10000, name='Equity Curve', line=dict(color='gold', width=3)), row=4, col=1)

fig.update_layout(height=1000, title_text="MACD Money Map Strategy - Full Backtest 2017–2025 (BTC/USDT 1H)", xaxis_rangeslider_visible=False)
fig.write_html("MACD_Money_Map_BTC_1H_2017_2025.html")
fig.write_image("MACD_Money_Map_BTC_1H_2017_2025.png")
fig.show()

print("\nInteractive chart saved as: MACD_Money_Map_BTC_1H_2017_2025.html")
print("Strategy fully implemented as per the original video — Zero Line Law + Distance Rule + Divergence + Triple Confirmation")