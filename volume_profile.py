import numpy as np
import pandas as pd

def compute_volume_profile(df_window, bin_percent=0.005, num_bins=None):
    """
    Compute Volume Profile for a given window of OHLCV data.
    Returns POC, VAH, VAL, HVN, LVN, and normalized heatmap.
    """
    if df_window.empty:
        return {'poc': 0, 'vah': 0, 'val': 0, 'hvn': [], 'lvn': [], 'heatmap': np.zeros(100)}

    prices = df_window[['high', 'low']].values.flatten()
    min_p = np.min(prices)
    max_p = np.max(prices)

    if min_p == max_p:
        poc = min_p
        vah = min_p
        val = min_p
        hvn = [min_p]
        lvn = [min_p]
        heatmap = np.array([1.0])
    else:
        avg_p = (min_p + max_p) / 2
        if num_bins:
            bin_size = (max_p - min_p) / num_bins
        else:
            bin_size = bin_percent * avg_p
        bins = np.arange(min_p, max_p + bin_size, bin_size)

        vp = np.zeros(len(bins) - 1)
        total_vol = df_window['volume'].sum()

        for _, row in df_window.iterrows():
            low = row['low']
            high = row['high']
            vol = row['volume']
            low_idx = int((low - min_p) / bin_size)
            high_idx = int((high - min_p) / bin_size)
            low_idx = max(0, min(low_idx, len(vp)-1))
            high_idx = max(0, min(high_idx, len(vp)-1))
            if low_idx <= high_idx:
                num_bins_touched = high_idx - low_idx + 1
                vol_per_bin = vol / num_bins_touched
                vp[low_idx:high_idx+1] += vol_per_bin

        # Normalize heatmap to fixed size 100 for consistency
        if len(vp) > 100:
            heatmap = np.interp(np.linspace(0, len(vp)-1, 100), np.arange(len(vp)), vp)
        elif len(vp) < 100:
            heatmap = np.pad(vp, (0, 100 - len(vp)), mode='constant')
        else:
            heatmap = vp
        if np.sum(heatmap) > 0:
            heatmap = heatmap / np.sum(heatmap)

        # POC
        poc_idx = np.argmax(vp)
        poc = bins[poc_idx] + bin_size / 2

        # VA: 70% volume around POC
        # Sort by volume desc
        sorted_indices = np.argsort(vp)[::-1]
        cum_vol = 0
        va_bins = []
        for idx in sorted_indices:
            cum_vol += vp[idx]
            va_bins.append(idx)
            if cum_vol >= 0.7 * total_vol:
                break
        va_bins = sorted(va_bins)
        if va_bins:
            val = bins[va_bins[0]]
            vah = bins[va_bins[-1]] + bin_size
        else:
            val = poc
            vah = poc

        # HVN: bins with volume > 75th percentile
        threshold = np.percentile(vp, 75)
        hvn_indices = np.where(vp > threshold)[0]
        hvn = bins[hvn_indices] + bin_size / 2

        # LVN: bins with volume < 25th percentile
        threshold_lvn = np.percentile(vp, 25)
        lvn_indices = np.where(vp < threshold_lvn)[0]
        lvn = bins[lvn_indices] + bin_size / 2

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'hvn': hvn,
        'lvn': lvn,
        'heatmap': heatmap,
        'max_volume': np.max(vp) if len(vp) > 0 else 0,
        'bins': bins,
        'vp': vp
    }

def get_rolling_vp(df, window_days, bin_percent=0.005):
    """
    Compute rolling Volume Profile for each timestamp.
    Returns a DataFrame with VP features.
    """
    window_hours = window_days * 24
    results = []
    total_iterations = len(df) - window_hours
    print(f"Computing rolling VP for {window_days} days ({window_hours} hours window), {total_iterations} iterations...")
    for i in range(window_hours, len(df)):
        if (i - window_hours) % 1000 == 0:
            print(f"VP {window_days}d: processed {i - window_hours}/{total_iterations} windows")
        window_df = df.iloc[i-window_hours:i]
        vp = compute_volume_profile(window_df, bin_percent)
        results.append(vp)
    # Pad the beginning
    empty_vp = {'poc': np.nan, 'vah': np.nan, 'val': np.nan, 'hvn': [], 'lvn': [], 'heatmap': np.zeros(100), 'max_volume': 0, 'bins': np.array([]), 'vp': np.array([])}
    results = [empty_vp] * (window_hours) + results
    vp_df = pd.DataFrame(results, index=df.index)
    return vp_df