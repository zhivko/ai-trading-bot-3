import numpy as np
import pandas as pd
import os
import pickle
import hashlib

def generate_cache_filename(df, window_days, bin_percent, cache_dir="vp_cache"):
    """
    Generates a unique filename based on data signature and parameters.
    """
    # 1. Create a signature of the data (Start time, End time, Length)
    # This ensures if you switch from BTC to ETH or update the CSV, the cache breaks.
    if isinstance(df.index, pd.DatetimeIndex):
        start_str = str(df.index[0])
        end_str = str(df.index[-1])
    else:
        # Fallback if no datetime index
        start_str = str(df.iloc[0].name)
        end_str = str(df.iloc[-1].name)
        
    data_signature = f"{start_str}_{end_str}_{len(df)}"
    
    # 2. Add parameters to signature
    param_signature = f"win{window_days}_bin{bin_percent}"
    
    # 3. Create Hash
    raw_string = f"{data_signature}_{param_signature}"
    file_hash = hashlib.md5(raw_string.encode('utf-8')).hexdigest()
    
    # 4. Construct Filename
    filename = f"vp_win{window_days}_{file_hash}.pkl"
    return os.path.join(cache_dir, filename)

def compute_volume_profile(df_window, bin_percent=0.005, num_bins=None):
    """
    Compute Volume Profile for a given window of OHLCV data.
    """
    if df_window.empty:
        return {'poc': 0, 'vah': 0, 'val': 0, 'heatmap': np.zeros(100)}

    prices = df_window[['high', 'low']].values.flatten()
    min_p = np.min(prices)
    max_p = np.max(prices)

    if min_p == max_p:
        return {'poc': min_p, 'vah': min_p, 'val': min_p, 'heatmap': np.zeros(100)}

    avg_p = (min_p + max_p) / 2
    if num_bins:
        bin_size = (max_p - min_p) / num_bins
    else:
        bin_size = bin_percent * avg_p
    
    # Safety check for bin_size
    if bin_size == 0: bin_size = 1.0
        
    bins = np.arange(min_p, max_p + bin_size, bin_size)
    vp = np.zeros(len(bins) - 1)
    total_vol = df_window['volume'].sum()

    for _, row in df_window.iterrows():
        low, high, vol = row['low'], row['high'], row['volume']
        low_idx = int((low - min_p) / bin_size)
        high_idx = int((high - min_p) / bin_size)
        low_idx = max(0, min(low_idx, len(vp)-1))
        high_idx = max(0, min(high_idx, len(vp)-1))
        
        if low_idx <= high_idx:
            num_bins_touched = high_idx - low_idx + 1
            vol_per_bin = vol / num_bins_touched
            vp[low_idx:high_idx+1] += vol_per_bin

    # Normalize heatmap to size 100
    target_size = 100
    if len(vp) != target_size:
        x_old = np.linspace(0, len(vp)-1, len(vp))
        x_new = np.linspace(0, len(vp)-1, target_size)
        heatmap = np.interp(x_new, x_old, vp)
    else:
        heatmap = vp
        
    if np.sum(heatmap) > 0:
        heatmap = heatmap / np.sum(heatmap)

    # POC
    poc_idx = np.argmax(vp)
    poc = bins[poc_idx] + bin_size / 2

    # VA Logic
    sorted_indices = np.argsort(vp)[::-1]
    cum_vol = 0
    va_bins = []
    for idx in sorted_indices:
        cum_vol += vp[idx]
        va_bins.append(idx)
        if cum_vol >= 0.7 * total_vol:
            break
            
    if va_bins:
        va_bins = sorted(va_bins)
        val = bins[va_bins[0]]
        vah = bins[va_bins[-1]] + bin_size
    else:
        val = poc
        vah = poc

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'heatmap': heatmap
    }

def get_rolling_vp(df, window_days, bin_percent=0.005, cache_dir="vp_cache"):
    """
    Compute rolling Volume Profile with Disk Caching (Pickle).
    """
    # 1. Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # 2. Check for Cache
    cache_path = generate_cache_filename(df, window_days, bin_percent, cache_dir)
    
    if os.path.exists(cache_path):
        print(f"  [Cache] Found existing Volume Profile: {cache_path}")
        print(f"  [Cache] Loading... (this is fast)")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [Cache] Error loading cache ({e}). Re-calculating.")

    # 3. Calculate if no cache
    window_hours = window_days * 24
    total_iterations = len(df) - window_hours
    
    print(f"  [Calc] Calculating rolling VP for {window_days} Days ({total_iterations} steps)...")
    
    pocs = np.zeros(len(df), dtype=np.float32)
    vahs = np.zeros(len(df), dtype=np.float32)
    vals = np.zeros(len(df), dtype=np.float32)
    heatmaps = np.zeros((len(df), 100), dtype=np.float32)
    
    for i in range(window_hours, len(df)):
        if (i - window_hours) % 2000 == 0:
            print(f"    Processed {i - window_hours}/{total_iterations}")
            
        window_df = df.iloc[i-window_hours:i]
        vp_data = compute_volume_profile(window_df, bin_percent)
        
        pocs[i] = vp_data['poc']
        vahs[i] = vp_data['vah']
        vals[i] = vp_data['val']
        heatmaps[i] = vp_data['heatmap']

    result = {
        'poc': pocs,
        'vah': vahs,
        'val': vals,
        'heatmap': heatmaps
    }
    
    # 4. Save to Cache
    print(f"  [Cache] Saving result to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
        
    return result