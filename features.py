import numpy as np
import pandas as pd

def get_features(df, vp7_df, vp30_df, t):
    """
    Extract features for a given timestamp t.
    """
    if t not in df.index or pd.isna(df.loc[t, 'close']):
        return np.zeros(200 + 24 + 8)  # Adjust size

    close = df.loc[t, 'close']

    # 7d VP
    vp7 = vp7_df.loc[t]
    poc7 = vp7['poc'] if not pd.isna(vp7['poc']) else close
    vah7 = vp7['vah'] if not pd.isna(vp7['vah']) else close
    val7 = vp7['val'] if not pd.isna(vp7['val']) else close
    hvn7 = vp7['hvn'] if isinstance(vp7['hvn'], list) else []
    lvn7 = vp7['lvn'] if isinstance(vp7['lvn'], list) else []
    heatmap7 = vp7['heatmap'] if isinstance(vp7['heatmap'], np.ndarray) else np.zeros(100)

    dist_hvn7 = min(abs(close - h) for h in hvn7) / close if hvn7 else 0
    dist_lvn7 = min(abs(close - l) for l in lvn7) / close if lvn7 else 0
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

    dist_hvn30 = min(abs(close - h) for h in hvn30) / close if hvn30 else 0
    dist_lvn30 = min(abs(close - l) for l in lvn30) / close if lvn30 else 0
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
        [dist_hvn7, dist_lvn7, rel_poc7, in_va7,  # 4
         dist_hvn30, dist_lvn30, rel_poc30, in_va30,  # 4
         vol, imbalance],  # 2
        session  # 24
    ])
    return features