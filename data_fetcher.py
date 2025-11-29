import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import argparse

PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAME = '1h'

def fetch_historical_ohlcv(pair, start_date, end_date=None):
    """
    Fetch historical OHLCV data from Binance for a given pair and date range.
    Handles pagination to get all data.
    """
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })

    if end_date is None:
        end_date = datetime.utcnow()

    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_data = []
    since = start_ts

    while since < end_ts:
        total_span = end_ts - start_ts
        current_progress = since - start_ts
        percentage = (current_progress / total_span) * 100 if total_span > 0 else 100
        print(f"Fetching {pair}: {percentage:.2f}% complete")
        try:
            data = exchange.fetch_ohlcv(pair, TIMEFRAME, since, 1000)
            if not data:
                break
            all_data.extend(data)
            since = data[-1][0] + 1  # Next timestamp
            time.sleep(1)  # Respect rate limit
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] <= end_date]
    df.set_index('timestamp', inplace=True)
    return df

def fetch_data_for_pairs(pairs, start_date, end_date=None):
    """
    Fetch data for multiple pairs.
    Returns a dict of DataFrames.
    """
    data = {}
    for pair in pairs:
        print(f"Fetching data for {pair}...")
        data[pair] = fetch_historical_ohlcv(pair, start_date, end_date)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical OHLCV data for a trading pair.')
    parser.add_argument('--pair', type=str, default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    args = parser.parse_args()
    pair = args.pair

    # Fetch data from 2021 to 2025
    start_date = '2016-01-01'
    end_date = '2025-11-29'
    print(f"Fetching data for {pair} from {start_date} to {end_date}...")
    df = fetch_historical_ohlcv(pair, start_date, end_date)
    filename = f'{pair.replace("/", "_")}_data.csv'
    df.to_csv(filename)
    print(f"Saved {len(df)} rows to {filename}")