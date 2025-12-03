import os
import glob
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from stable_baselines3 import PPO, SAC
import threading

# Import your custom environment
from enhanced_trading_env import EnhancedTradingEnv

app = Flask(__name__)
socketio = SocketIO(app)

CRYPTO_PAIR = "BTCUSDT"

# --- CONFIGURATION ---
MODEL_PATH_WILDCARD = [
    f"./models/sac_{CRYPTO_PAIR}_final.zip",
    f"./models/{CRYPTO_PAIR}_best_eval/*_{CRYPTO_PAIR}_*.zip",
    "./logs/best_model.zip",
    f"./models/sac_{CRYPTO_PAIR}.zip"
]
DATA_PATH = f"{CRYPTO_PAIR}_data.csv"

# THREAD-SAFE GLOBAL MODEL CACHE
_model_cache = {}
_model_lock = threading.Lock()

def find_model_once():
    """Find model ONCE, share across ALL threads/workers"""
    global MODEL_PATH, ALGORITHM
    with _model_lock:
        if MODEL_PATH is None:
            for pattern in MODEL_PATH_WILDCARD:
                model_files = glob.glob(pattern) if '*' in pattern else [pattern] if os.path.exists(pattern) else []
                if model_files:
                    MODEL_PATH = model_files[0]
                    print(f"🔍 Found model (shared): {MODEL_PATH}")
                    ALGORITHM = "SAC" if "sac" in MODEL_PATH.lower() else "PPO"
                    break
    return MODEL_PATH, ALGORITHM

# Initialize globals for thread safety
MODEL_PATH = None
ALGORITHM = None
TEST_SPLIT = None

# Find model once (thread-safe)
MODEL_PATH, ALGORITHM = find_model_once()

# Store results globally
GLOBAL_RESULTS = None
current_start = None
current_end = None

def load_data():
    """Loads and prepares data."""
    print(f"Reading {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    
    # Handle Binance raw data format if needed
    if 'open_time' in df.columns and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])
    else:
        df['date'] = pd.to_datetime(df['date'])
        
    # Unify column name to timestamp for internal consistency
    df['timestamp'] = df['date']
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Filter data from test split date to last entry
    test_split_date = pd.to_datetime(TEST_SPLIT)
    df = df[df['timestamp'] >= test_split_date].reset_index(drop=True)

    return df

def run_simulation():
    """Runs the simulation ONCE at startup."""
    if not MODEL_PATH:
        print("❌ No model found, skipping simulation.")
        return None

    print("--- STARTING BACKTEST SIMULATION ---")

    # Load metadata from model to get test_split
    try:
        model_temp, metadata = SAC.load(MODEL_PATH, return_metadata=True)
        TEST_SPLIT = metadata.get("test_split", "2023-01-01")
        print(f"Loaded model metadata: {metadata}")
    except Exception as e:
        print(f"Could not load model metadata: {e}. Using default test_split.")
        TEST_SPLIT = "2023-01-01"

    df = load_data()
    
    print("Initializing Environment (Forcing vp_bins=40)...")
    # CRITICAL: Use EnhancedTradingEnv to match trained model (438 dims)
    env = EnhancedTradingEnv(df, initial_balance=1000, lookback_window=50, vp_bins=40, vp_days=[7, 30])
    
    print(f"Loading Model from {MODEL_PATH}...")
    
    try:
        # Force SAC since the error indicates SAC model
        model = SAC.load(MODEL_PATH, custom_objects={'use_sde': False})
        model.set_env(env)
        print(f"Model loaded successfully. Observation space: {model.observation_space.shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    print("Running Loop...")
    obs, _ = env.reset()
    done = False
    history = []
    total_steps = len(df)
    last_progress = 0

    # Run full history
    while not done:
        # 1. Get deterministic prediction
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. --- CRITICAL FIX: ACTION THRESHOLD ---
        # Prevents overtrading on weak signals (noise)
        # If the agent isn't confident (action < 0.15), we force a HOLD (0)
        if abs(action[0]) < 0.15:
            action[0] = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_data = {
            "timestamp": info.get("date") or info.get("timestamp"),
            "date": info.get("date") or info.get("timestamp"),
            "close": info.get('price'),
            "net_worth": info.get('portfolio_value'),
            "balance_usdt": info.get('balance'),
            "shares_held": info.get('shares_held'),
            "action": float(action[0]),
            "open": df.iloc[env.current_step]['open'],
            "high": df.iloc[env.current_step]['high'],
            "low": df.iloc[env.current_step]['low'],
        }
        history.append(step_data)

        # Progress output
        progress = int((env.current_step / total_steps) * 100)
        if progress > last_progress:
            print(f"Simulation progress: {progress}%")
            last_progress = progress

    print("--- SIMULATION COMPLETE ---")
    if history:
        df_hist = pd.DataFrame(history)
        start_timestamp = df_hist['timestamp'].iloc[0]
        end_timestamp = df_hist['timestamp'].iloc[-1]
        initial_balance = 1000
        final_balance = df_hist['net_worth'].iloc[-1]
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        cumulative = df_hist['net_worth']
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        actions = df_hist['action']
        # Count only actual trades (thresholded)
        buys = (actions > 0).sum()
        sells = (actions < 0).sum()
        num_trades = buys + sells
        
        print(f"Simulation Period: {start_timestamp} to {end_timestamp}")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Steps: {len(history)}")
        print(f"Number of Trades: {num_trades}")
        print(f"Buys: {buys}, Sells: {sells}")
        
        return pd.DataFrame(history)
    return None

def create_plot(df, start_timestamp=None, end_timestamp=None, include_plotlyjs=True):
    """Creates the Plotly interactive chart."""
    if start_timestamp or end_timestamp:
        df = df.copy()
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date' if 'date' in df.columns else None
        if time_col and start_timestamp:
            df = df[df[time_col] >= pd.to_datetime(start_timestamp)]
        if time_col and end_timestamp:
            df = df[df[time_col] <= pd.to_datetime(end_timestamp)]
    
    if df.empty:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(title="No Data Selected")
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id='chart')

    # Resample to daily to reduce data points for plotting if dataset is huge
    time_col = 'timestamp' if 'timestamp' in df.columns else 'date' if 'date' in df.columns else None
    
    # NOTE: You can adjust resampling rule to '4h' or '1h' if you want more detail on zoom
    if time_col:
        df = df.set_index(time_col)
        df_daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'action': 'mean', # Average action intensity for the day
            'net_worth': 'last',
            'balance_usdt': 'last',
            'shares_held': 'last'
        }).dropna()
        df = df_daily.reset_index()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=("Price Action & Trades", "Net Worth", "USDT", "BTC")
    )

    # 1. Price
    fig.add_trace(go.Candlestick(
        x=df[time_col], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)

    # Markers (Simple aggregation for visualization)
    # Note: On daily aggregation, action is mean, so > 0.05 suggests Buying pressure
    buys = df[df['action'] > 0.05]
    fig.add_trace(go.Scatter(
        x=buys[time_col], y=buys['close'], mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10), name='Buy',
        text=[f"Shares: {s:.6f}" for s in buys['shares_held']],
        hovertemplate='%{text}<br>Price: %{y}<extra></extra>'
    ), row=1, col=1)

    sells = df[df['action'] < -0.05]
    fig.add_trace(go.Scatter(
        x=sells[time_col], y=sells['close'], mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10), name='Sell',
        text=[f"Shares: {s:.6f}" for s in sells['shares_held']],
        hovertemplate='%{text}<br>Price: %{y}<extra></extra>'
    ), row=1, col=1)

    # 2. Net Worth
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df['net_worth'], line=dict(color='#00bfff', width=2), name='Net Worth'
    ), row=2, col=1)

    # 3. USDT
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df['balance_usdt'], line=dict(color='#00ff00', width=1), fill='tozeroy', name='USDT'
    ), row=3, col=1)

    # 4. BTC
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df['shares_held'], line=dict(color='#ffa500', width=1), fill='tozeroy', name='BTC'
    ), row=4, col=1)

    # Vertical lines for hover (one per subplot)
    for row in [1, 2, 3, 4]:
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines', line=dict(color='white', width=1),
            showlegend=False, hoverinfo='skip'
        ), row=row, col=1)

    fig.update_layout(
        title=f"AI Bot Backtest Results",
        height=1200,
        template="plotly_dark",
        hovermode="x",
        dragmode='pan',
        margin=dict(l=40, r=40, t=70, b=40),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(spikemode='across', spikesnap='cursor', showspikes=True)
    fig.update_yaxes(autorange=True)

    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id='chart')

def get_trace_data(df):
    """Prepares trace data for websocket updates."""
    timestamps_iso = [d.isoformat() for d in df['timestamp']]
    candlestick = {
        'x': timestamps_iso,
        'open': df['open'].tolist(),
        'high': df['high'].tolist(),
        'low': df['low'].tolist(),
        'close': df['close'].tolist(),
        'type': 'candlestick',
        'name': 'Price',
        'xaxis': 'x',
        'yaxis': 'y'
    }

    buys = df[df['action'] > 0.05]
    buys_data = {
        'x': [d.isoformat() for d in buys['timestamp']],
        'y': buys['close'].tolist(),
        'mode': 'markers',
        'marker': {'symbol': 'triangle-up', 'color': 'green', 'size': 12},
        'name': 'Buy',
        'xaxis': 'x',
        'yaxis': 'y',
        'text': [f"Shares: {s:.6f}" for s in buys['shares_held']],
        'hovertemplate': '%{text}<br>Price: %{y}<extra></extra>'
    }

    sells = df[df['action'] < -0.05]
    sells_data = {
        'x': [d.isoformat() for d in sells['timestamp']],
        'y': sells['close'].tolist(),
        'mode': 'markers',
        'marker': {'symbol': 'triangle-down', 'color': 'red', 'size': 12},
        'name': 'Sell',
        'xaxis': 'x',
        'yaxis': 'y',
        'text': [f"Shares: {s:.6f}" for s in sells['shares_held']],
        'hovertemplate': '%{text}<br>Price: %{y}<extra></extra>'
    }

    net_worth = {
        'x': timestamps_iso,
        'y': df['net_worth'].tolist(),
        'line': {'color': '#00bfff', 'width': 2},
        'name': 'Net Worth',
        'xaxis': 'x2',
        'yaxis': 'y2'
    }

    balance = {
        'x': timestamps_iso,
        'y': df['balance_usdt'].tolist(),
        'line': {'color': '#00ff00', 'width': 1},
        'fill': 'tozeroy',
        'name': 'USDT',
        'xaxis': 'x3',
        'yaxis': 'y3'
    }

    shares = {
        'x': timestamps_iso,
        'y': df['shares_held'].tolist(),
        'line': {'color': '#ffa500', 'width': 1},
        'fill': 'tozeroy',
        'name': 'BTC',
        'xaxis': 'x4',
        'yaxis': 'y4'
    }

    # Vertical lines (empty initially)
    vline1 = {'x': [], 'y': [], 'mode': 'lines', 'line': {'color': 'white', 'width': 1}, 'showlegend': False, 'hoverinfo': 'skip', 'xaxis': 'x', 'yaxis': 'y'}
    vline2 = {'x': [], 'y': [], 'mode': 'lines', 'line': {'color': 'white', 'width': 1}, 'showlegend': False, 'hoverinfo': 'skip', 'xaxis': 'x2', 'yaxis': 'y2'}
    vline3 = {'x': [], 'y': [], 'mode': 'lines', 'line': {'color': 'white', 'width': 1}, 'showlegend': False, 'hoverinfo': 'skip', 'xaxis': 'x3', 'yaxis': 'y3'}
    vline4 = {'x': [], 'y': [], 'mode': 'lines', 'line': {'color': 'white', 'width': 1}, 'showlegend': False, 'hoverinfo': 'skip', 'xaxis': 'x4', 'yaxis': 'y4'}

    layout = {
        'title': 'AI Bot Backtest Results',
        'height': 1200,
        'template': 'plotly_dark',
        'hovermode': 'x',
        'dragmode': 'pan',
        'margin': {'l': 40, 'r': 40, 't': 70, 'b': 40},
        'paper_bgcolor': '#111111',
        'plot_bgcolor': '#111111',
        'xaxis': {'type': 'date', 'range': [timestamps_iso[0], timestamps_iso[-1]], 'domain': [0, 1], 'rangeslider': {'visible': False}, 'spikemode': 'across', 'spikesnap': 'cursor', 'showspikes': True},
        'yaxis': {'domain': [0.5, 1], 'autorange': True},
        'xaxis2': {'matches': 'x', 'showticklabels': False, 'domain': [0, 1]},
        'yaxis2': {'domain': [0.35, 0.5], 'autorange': True},
        'xaxis3': {'matches': 'x', 'showticklabels': False, 'domain': [0, 1]},
        'yaxis3': {'domain': [0.15, 0.35], 'autorange': True},
        'xaxis4': {'matches': 'x', 'showticklabels': False, 'domain': [0, 1]},
        'yaxis4': {'domain': [0, 0.15], 'autorange': True}
    }

    return {
        'traces': [candlestick, buys_data, sells_data, net_worth, balance, shares, vline1, vline2, vline3, vline4],
        'layout': layout
    }

@app.route('/data.csv')
def serve_csv():
    global GLOBAL_RESULTS
    if GLOBAL_RESULTS is None:
        try:
            with open('backtest_results.pkl', 'rb') as f:
                GLOBAL_RESULTS = pickle.load(f)
        except FileNotFoundError:
            pass
    if GLOBAL_RESULTS is None or GLOBAL_RESULTS.empty:
        return "No data available", 404

    csv_data = GLOBAL_RESULTS.to_csv(index=False)
    return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=backtest_data.csv'}

@app.route('/')
def index():
    global GLOBAL_RESULTS
    if GLOBAL_RESULTS is None:
        try:
            with open('backtest_results.pkl', 'rb') as f:
                GLOBAL_RESULTS = pickle.load(f)
        except FileNotFoundError:
            pass
    if GLOBAL_RESULTS is None or GLOBAL_RESULTS.empty:
        return "<h1>Simulation Failed or Returned No Data. Check Console.</h1>"

    # Initially show last 5 months
    if 'timestamp' in GLOBAL_RESULTS.columns:
        max_timestamp = GLOBAL_RESULTS['timestamp'].max()
    else:
        max_timestamp = pd.Timestamp.now()
    
    global current_start, current_end
    current_end = max_timestamp
    current_start = max_timestamp - pd.DateOffset(months=5)
    
    initial = GLOBAL_RESULTS.iloc[0]['net_worth']
    final = GLOBAL_RESULTS.iloc[-1]['net_worth']
    roi = ((final - initial) / initial) * 100 if initial > 0 else 0

    html_chart = create_plot(GLOBAL_RESULTS, start_timestamp=current_start, end_timestamp=current_end)

    return render_template('index.html', chart=html_chart, initial=initial, final=final, roi=roi)

@socketio.on('range_change')
def handle_range(data):
    global current_start, current_end, GLOBAL_RESULTS
    if GLOBAL_RESULTS is None:
        try:
            with open('backtest_results.pkl', 'rb') as f:
                GLOBAL_RESULTS = pickle.load(f)
        except FileNotFoundError:
            return

    start = pd.to_datetime(data.get('start'))
    end = pd.to_datetime(data.get('end'))
    current_start = start
    current_end = end

    df_filtered = GLOBAL_RESULTS[(GLOBAL_RESULTS['timestamp'] >= current_start) & (GLOBAL_RESULTS['timestamp'] <= current_end)]

    if df_filtered.empty:
        return

    df = df_filtered.set_index('timestamp')
    df_daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'action': 'mean',
        'net_worth': 'last',
        'balance_usdt': 'last',
        'shares_held': 'last'
    }).dropna()
    df = df_daily.reset_index()

    data_dict = get_trace_data(df)
    data_dict['start'] = current_start.isoformat()
    data_dict['end'] = current_end.isoformat()
    emit('update_traces', data_dict)

@socketio.on('pan')
def handle_pan(data):
    global current_start, current_end, GLOBAL_RESULTS
    if GLOBAL_RESULTS is None:
        try:
            with open('backtest_results.pkl', 'rb') as f:
                GLOBAL_RESULTS = pickle.load(f)
        except FileNotFoundError:
            return

    direction = data.get('direction')
    delta = pd.DateOffset(months=1)

    if direction == 'left':
        current_start -= delta
        current_end -= delta
    elif direction == 'right':
        current_start += delta
        current_end += delta

    df_filtered = GLOBAL_RESULTS[(GLOBAL_RESULTS['timestamp'] >= current_start) & (GLOBAL_RESULTS['timestamp'] <= current_end)]

    if df_filtered.empty:
        return

    df = df_filtered.set_index('timestamp')
    df_daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'action': 'mean',
        'net_worth': 'last',
        'balance_usdt': 'last',
        'shares_held': 'last'
    }).dropna()
    df = df_daily.reset_index()

    data_dict = get_trace_data(df)
    data_dict['start'] = current_start.isoformat()
    data_dict['end'] = current_end.isoformat()
    emit('update_traces', data_dict)

if __name__ == "__main__":
    # RUN SIMULATION ONCE IN MAIN THREAD (only in parent process, not on reload)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        GLOBAL_RESULTS = run_simulation()
        # Save results to file for child processes
        if GLOBAL_RESULTS is not None:
            with open('backtest_results.pkl', 'wb') as f:
                pickle.dump(GLOBAL_RESULTS, f)

    print("Starting Server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=False, use_reloader=False)
