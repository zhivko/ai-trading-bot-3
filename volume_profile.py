import numpy as np
import pandas as pd
import os
import hashlib
import pickle
import multiprocessing
from functools import partial

# Configuration
CACHE_DIR = "vp_cache"

def generate_cache_filename(df, window_days, num_bins, cache_dir="vp_cache"):
    """
    Generates a unique filename based on data hash and parameters.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Hash the critical columns to ensure data integrity
    data_bytes = df[['close', 'volume']].values.tobytes()
    data_hash = hashlib.md5(data_bytes).hexdigest()
    
    filename = f"vp_win{window_days}_bins{num_bins}_{data_hash}.pkl"
    return os.path.join(cache_dir, filename)

def calculate_vp(prices, volumes, bins=40):
    """
    Calculates Volume Profile, including POC, VAH, VAL, HVN, and LVN.
    """
    if len(prices) == 0:
        return 0.0, 0.0, 0.0, [], [], np.zeros(bins)
        
    min_p = np.min(prices)
    max_p = np.max(prices)
    
    if min_p == max_p:
        return min_p, min_p, min_p, [], [], np.zeros(bins)
        
    hist, bin_edges = np.histogram(
        prices, 
        bins=bins, 
        range=(min_p, max_p), 
        weights=volumes, 
        density=False
    )
    
    # Normalize Heatmap (0 to 1)
    if np.max(hist) > 0:
        hist_norm = hist / np.max(hist)
    else:
        hist_norm = hist

    # POC (Point of Control)
    max_idx = np.argmax(hist)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
    
    # VAH/VAL (70% Value Area)
    total_vol = np.sum(hist)
    target_vol = total_vol * 0.70
    current_vol = hist[max_idx]
    left, right = max_idx, max_idx
    
    while current_vol < target_vol:
        v_left = hist[left - 1] if left > 0 else 0
        v_right = hist[right + 1] if right < len(hist) - 1 else 0
        
        if v_left == 0 and v_right == 0:
            break
            
        if v_left > v_right:
            left -= 1
            current_vol += v_left
        else:
            right += 1
            current_vol += v_right
            
    val_price = bin_edges[left]
    vah_price = bin_edges[right + 1]

    # --- HVN / LVN Calculation ---
    hvns = []
    lvns = []
    
    hist_padded = np.pad(hist, (1, 1), 'constant', constant_values=0)
    for i in range(1, len(hist_padded) - 1):
        prev_val = hist_padded[i-1]
        curr_val = hist_padded[i]
        next_val = hist_padded[i+1]
        price = (bin_edges[i-1] + bin_edges[i]) / 2
        
        if curr_val > prev_val and curr_val > next_val:
            hvns.append(price)
        elif curr_val < prev_val and curr_val < next_val and curr_val > 0:
            lvns.append(price)

    return poc_price, vah_price, val_price, hvns, lvns, hist_norm

def _calculate_single_step_vp(idx, prices, volumes, lookback_days, bins=40):
    """
    Worker function for multiprocessing. 
    """
    window_len = lookback_days * 24
    start_idx = max(0, idx - window_len)
    end_idx = idx
    current_window_prices = prices[start_idx:end_idx]
    current_window_vols = volumes[start_idx:end_idx]
    return calculate_vp(current_window_prices, current_window_vols, bins)

def get_rolling_vp(df, days, bins=40):
    """
    Main entry point for Training.
    Returns dictionary with: poc, vah, val, hvn, lvn, heatmap
    """
    filepath = generate_cache_filename(df, days, bins, CACHE_DIR)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Integrity check
            if len(data['heatmap']) > 0 and len(data['heatmap'][0]) == bins:
                if 'hvn' in data: # Check for restored keys
                    print(f"⚡ [VP] Loaded cached {days}d profile (Bins: {bins})")
                    return data
        except Exception as e:
            print(f"⚠️ [VP] Cache read error: {e}")

    print(f"⚙️ [VP] Calculating Rolling VP for {days} days (Bins: {bins})...")
    
    raw_prices = df['close'].values
    raw_volumes = df['volume'].values
    indices = range(len(df))
    
    num_cores = min(multiprocessing.cpu_count(), 8) 
    
    worker = partial(
        _calculate_single_step_vp, 
        prices=raw_prices, 
        volumes=raw_volumes, 
        lookback_days=days, 
        bins=bins
    )
    
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(worker, indices)
    
    pocs, vahs, vals, hvns, lvns, heatmaps = zip(*results)
    
    vp_data = {
        'poc': np.array(pocs),
        'vah': np.array(vahs),
        'val': np.array(vals),
        'hvn': np.array(hvns, dtype=object),
        'lvn': np.array(lvns, dtype=object),
        'heatmap': np.array(heatmaps)
    }
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(vp_data, f)
        print(f"💾 [VP] Saved cache: {filepath}")
    except Exception as e:
        print(f"⚠️ [VP] Could not save cache: {e}")
        
    return vp_data