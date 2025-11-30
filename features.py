import numpy as np
import pandas as pd
import ta
import scipy.stats

def get_features(df, vp7_df, vp30_df, t):
    """
    Extract features for a given timestamp t.
    """
    if t not in df.index or pd.isna(df.loc[t, 'close']):
        return np.zeros(259)  # Updated size with heatmap features added


    close = df.loc[t, 'close']

    # Technical Indicators
    data_up_to_t = df.loc[:t, 'close']
    if len(data_up_to_t) >= 26:  # MACD needs at least 26
        macd_indicator = ta.trend.MACD(data_up_to_t, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_indicator.macd().iloc[-1] if not pd.isna(macd_indicator.macd().iloc[-1]) else 0
        macd_signal = macd_indicator.macd_signal().iloc[-1] if not pd.isna(macd_indicator.macd_signal().iloc[-1]) else 0
        macd_hist = macd_indicator.macd_diff().iloc[-1] if not pd.isna(macd_indicator.macd_diff().iloc[-1]) else 0
    else:
        macd_line, macd_signal, macd_hist = 0, 0, 0

    if len(data_up_to_t) >= 14:  # RSI needs at least 14
        rsi_indicator = ta.momentum.RSIIndicator(data_up_to_t, window=14)
        rsi = rsi_indicator.rsi().iloc[-1] if not pd.isna(rsi_indicator.rsi().iloc[-1]) else 50
    else:
        rsi = 50

    if len(data_up_to_t) >= 14:  # Stoch RSI needs at least 14
        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(data_up_to_t, window=14, smooth1=3, smooth2=3)
        stoch_k = stoch_rsi_indicator.stochrsi_k().iloc[-1] if not pd.isna(stoch_rsi_indicator.stochrsi_k().iloc[-1]) else 0.5
        stoch_d = stoch_rsi_indicator.stochrsi_d().iloc[-1] if not pd.isna(stoch_rsi_indicator.stochrsi_d().iloc[-1]) else 0.5
    else:
        stoch_k, stoch_d = 0.5, 0.5

    if len(data_up_to_t) >= 14:  # ATR needs at least 14
        high_up_to_t = df.loc[:t, 'high']
        low_up_to_t = df.loc[:t, 'low']
        close_up_to_t = df.loc[:t, 'close']
        atr_indicator = ta.volatility.AverageTrueRange(high_up_to_t, low_up_to_t, close_up_to_t, window=14)
        atr = atr_indicator.average_true_range().iloc[-1] if not pd.isna(atr_indicator.average_true_range().iloc[-1]) else 0
    else:
        atr = 0

    # 7d VP
    vp7 = vp7_df.loc[t]
    poc7 = vp7['poc'] if not pd.isna(vp7['poc']) else close
    vah7 = vp7['vah'] if not pd.isna(vp7['vah']) else close
    val7 = vp7['val'] if not pd.isna(vp7['val']) else close
    hvn7 = vp7['hvn'] if isinstance(vp7['hvn'], list) else []
    lvn7 = vp7['lvn'] if isinstance(vp7['lvn'], list) else []
    heatmap7 = vp7['heatmap'] if isinstance(vp7['heatmap'], np.ndarray) else np.zeros(100)

    # Normalized POC, VAH, VAL
    norm_poc7 = (poc7 / close) - 1
    norm_vah7 = (vah7 / close) - 1
    norm_val7 = (val7 / close) - 1

    # HVN/LVN statistics
    hvn_count7 = len(hvn7)
    lvn_count7 = len(lvn7)
    hvn_avg_dist7 = np.mean([abs(close - h) / close for h in hvn7]) if hvn7 else 0
    lvn_avg_dist7 = np.mean([abs(close - l) / close for l in lvn7]) if lvn7 else 0
    hvn_nearest7 = min(abs(close - h) / close for h in hvn7) if hvn7 else 0
    lvn_nearest7 = min(abs(close - l) / close for l in lvn7) if lvn7 else 0

    dist_hvn7 = hvn_nearest7  # Keep existing
    dist_lvn7 = lvn_nearest7  # Keep existing
    rel_poc7 = (close - poc7) / poc7 if poc7 != 0 else 0
    in_va7 = 1 if val7 <= close <= vah7 else 0

    # 30d VP
    vp30 = vp30_df.loc[t]
    poc30 = vp30['poc'] if not pd.isna(vp30['poc']) else close
    vah30 = vp30['vah'] if not pd.isna(vp30['vah']) else close
    val30 = vp30['val'] if not pd.isna(vp30['val']) else close
    hvn30 = vp30['hvn'] if isinstance(vp30['hvn'], list) else []
    lvn30 = vp30['lvn'] if isinstance(vp30['lvn'], list) else []
    heatmap30 = vp30['heatmap'] if isinstance(vp30['heatmap'], np.ndarray) else np.zeros(100)

    # Normalized POC, VAH, VAL
    norm_poc30 = (poc30 / close) - 1
    norm_vah30 = (vah30 / close) - 1
    norm_val30 = (val30 / close) - 1

    # HVN/LVN statistics
    hvn_count30 = len(hvn30)
    lvn_count30 = len(lvn30)
    hvn_avg_dist30 = np.mean([abs(close - h) / close for h in hvn30]) if hvn30 else 0
    lvn_avg_dist30 = np.mean([abs(close - l) / close for l in lvn30]) if lvn30 else 0
    hvn_nearest30 = min(abs(close - h) / close for h in hvn30) if hvn30 else 0
    lvn_nearest30 = min(abs(close - l) / close for l in lvn30) if lvn30 else 0

    dist_hvn30 = hvn_nearest30  # Keep existing
    dist_lvn30 = lvn_nearest30  # Keep existing
    rel_poc30 = (close - poc30) / poc30 if poc30 != 0 else 0
    in_va30 = 1 if val30 <= close <= vah30 else 0

    # Volatility: 7-day rolling std of returns
    returns = df['close'].pct_change()
    vol = returns.rolling('7D').std().loc[t] if t in returns.rolling('7D').std().index else 0
    if pd.isna(vol):
        vol = 0

    # Order-book imbalance placeholder (hard to get historical, use close-open / (high-low))
    row = df.loc[t]
    imbalance = (row['close'] - row['open']) / (row['high'] - row['low']) if row['high'] != row['low'] else 0

    # Session: one-hot for hour
    hour = t.hour
    session = np.zeros(24)
    session[hour] = 1

    # Combine features
    features = np.concatenate([
           heatmap7,  # 100
           heatmap30,  # 100
           [norm_poc7, norm_vah7, norm_val7],  # 3
           [norm_poc30, norm_vah30, norm_val30],  # 3
           [hvn_count7, hvn_avg_dist7, hvn_nearest7, lvn_count7, lvn_avg_dist7, lvn_nearest7],  # 6
           [hvn_count30, hvn_avg_dist30, hvn_nearest30, lvn_count30, lvn_avg_dist30, lvn_nearest30],  # 6
           [dist_hvn7, dist_lvn7, rel_poc7, in_va7,  # 4
            dist_hvn30, dist_lvn30, rel_poc30, in_va30,  # 4
            vol, imbalance],  # 2
           [macd_line, macd_signal, macd_hist, rsi, stoch_k, stoch_d, atr],  # 7
           session  # 24
       ])


    return features