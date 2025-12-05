import yfinance as yf
import pandas as pd
import ta
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# --- CONFIGURATION ---
TICKERS = ['BLK', 'BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'NVDA', 'AMD'] 
TIMEFRAME = "15m"   
# Slow Stoch (60, 10, 10) matches your chart style
STOCH_SETTINGS = [{"k": 60, "d": 10, "smooth": 10}] 

def find_peaks_divergence(df, stoch_series, lookback=60):
    """
    Uses localized extrema (valleys) to find the specific divergence 
    pattern matching the user's 'Yellow Line'.
    """
    # 1. Find all "Valleys" (Local Minima) in the Stochastic
    # order=3 means a point must be lower than 3 neighbors on left AND right
    # This filters out small jagged noise.
    stoch_vals = stoch_series.values
    minima_indices = argrelextrema(stoch_vals, np.less, order=3)[0]
    
    # Filter indices to be within our lookback window
    current_idx = len(df) - 1
    valid_minima = [i for i in minima_indices if i > (current_idx - lookback)]
    
    # We need at least 2 valleys to compare
    if len(valid_minima) < 2:
        return None, "Not enough valleys"

    # 2. Iterate through pairs of valleys to find the divergence
    # We look at the most recent valleys first
    # Let B be the recent valley, A be the older valley
    
    # We loop backwards to find the most relevant setup
    for i in range(len(valid_minima) - 1, 0, -1):
        idx_B = valid_minima[i]      # Recent Trough (Point B)
        idx_A = valid_minima[i - 1]  # Previous Trough (Point A)
        
        # If Point B is too old (happened 20+ bars ago), it's not a trade signal anymore
        if (current_idx - idx_B) > 20:
            continue

        # Get Values
        price_A = df['Low'].iloc[idx_A]
        price_B = df['Low'].iloc[idx_B]
        
        stoch_A = stoch_series.iloc[idx_A]
        stoch_B = stoch_series.iloc[idx_B]
        
        # --- THE GOLDEN RULE (Your Yellow Line) ---
        
        # 1. Price MUST be Lower (or roughly equal double bottom)
        # This fixes the "Pink Line" issue where price was rising.
        price_is_lower = price_B <= (price_A * 1.002) # Allow tiny tolerance (0.2%)
        
        # 2. Stoch MUST be Higher
        # This ensures momentum is building up.
        stoch_is_higher = stoch_B > (stoch_A + 2)
        
        # 3. Both must be Oversold (<50)
        in_zone = stoch_A < 50 and stoch_B < 50
        
        if price_is_lower and stoch_is_higher and in_zone:
            # Check if we are currently in a "Buy" position (Price bouncing up from B)
            # If current price is way below B, it's crashing. We want a hook.
            curr_price = df['Close'].iloc[-1]
            if curr_price >= price_B:
                return {
                    "idx_A": idx_A, "price_A": price_A, "stoch_A": stoch_A,
                    "idx_B": idx_B, "price_B": price_B, "stoch_B": stoch_B
                }, "MATCH"

    return None, "No divergence found"

def plot_divergence_chart(df, ticker, div_data, stoch_series):
    # Zoom window: Start 10 candles before Point A
    start_idx = max(0, div_data['idx_A'] - 10)
    subset = df.iloc[start_idx:].copy()
    subset_stoch = stoch_series.iloc[start_idx:].copy()
    
    # Coordinates relative to subset
    p1_idx = div_data['idx_A'] - start_idx
    p2_idx = div_data['idx_B'] - start_idx
    
    buy_idx = len(subset) - 1
    buy_price = subset['Close'].iloc[-1]

    # Plot Config
    apds = [ mpf.make_addplot(subset_stoch, color='#FFA500', panel=1, ylabel='Stoch(60)', width=2) ]
    
    filename = f"{ticker}_true_divergence.png"
    
    fig, axlist = mpf.plot(
        subset, type='candle', style='nightclouds',
        addplot=apds, volume=False, panel_ratios=(2, 1),
        title=f"\n{ticker} - TRUE DIVERGENCE (Yellow Line Pattern)",
        returnfig=True, figsize=(10, 8)
    )
    
    # --- DRAW LINES (YELLOW to match your drawing) ---
    
    # 1. Price Line (Sloping DOWN)
    axlist[0].plot(
        [p1_idx, p2_idx], 
        [div_data['price_A'], div_data['price_B']], 
        color='yellow', linewidth=3, marker='o'
    )

    # 2. Stoch Line (Sloping UP)
    axlist[2].plot(
        [p1_idx, p2_idx], 
        [div_data['stoch_A'], div_data['stoch_B']], 
        color='yellow', linewidth=3, marker='o'
    )
    
    # 3. Buy Circle
    axlist[0].plot(
        buy_idx, buy_price, 
        marker='o', markersize=25, 
        markerfacecolor='none', markeredgecolor='#00FF00', markeredgewidth=3
    )
    axlist[0].text(
        buy_idx, buy_price * 1.001, '  <-- BUY', 
        color='#00FF00', fontweight='bold', verticalalignment='center'
    )

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"   📷 Chart saved: {filename}")

def scan_divergence():
    print(f"Scanning for TRUE DIVERGENCE (Yellow Line Logic) on {TIMEFRAME}...")
    print("=" * 70)
    
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="5d", interval=TIMEFRAME, progress=False, auto_adjust=True)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.astype(float)

            # Calc Stoch
            s = STOCH_SETTINGS[0]
            stoch_k = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=s['k'], smooth_window=s['smooth'])

            div_data, msg = find_peaks_divergence(df, stoch_k, lookback=80)
            
            if div_data:
                print(f"✅ FOUND DIVERGENCE: {ticker}")
                print(f"   Price: {div_data['price_A']:.2f} -> {div_data['price_B']:.2f} (LOWER)")
                print(f"   Stoch: {div_data['stoch_A']:.2f} -> {div_data['stoch_B']:.2f} (HIGHER)")
                plot_divergence_chart(df, ticker, div_data, stoch_k)
            else:
                pass # Silent on misses
            
        except Exception as e:
            print(f"Error {ticker}: {e}")
    
    print("-" * 70)

if __name__ == "__main__":
    scan_divergence()