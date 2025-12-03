import os
import glob
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template_string, Response
from flask_socketio import SocketIO, emit
from stable_baselines3 import PPO, SAC
import threading

# Import your custom environment
from trading_env import TradingEnv

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

# Find model once (thread-safe)
MODEL_PATH, ALGORITHM = find_model_once()
if not MODEL_PATH:
    raise FileNotFoundError(f"No model files found. Searched patterns: {MODEL_PATH_WILDCARD}")

if not MODEL_PATH:
    raise FileNotFoundError(f"No model files found. Searched patterns: {MODEL_PATH_WILDCARD}")

# Store results globally
GLOBAL_RESULTS = None
current_start = None
current_end = None

def load_data():
    """Loads and prepares data."""
    print(f"Reading {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    path = DATA_PATH
        
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def run_simulation():
    """Runs the simulation ONCE at startup."""
    print("--- STARTING BACKTEST SIMULATION ---")
    df = load_data()
    
    print("Initializing Environment (Forcing vp_bins=40)...")
    # CRITICAL: vp_bins=40 makes observation space 388, matching your trained model.
    env = TradingEnv(df, initial_balance=1000, lookback_window=50, vp_bins=40, vp_days=[7, 30])
    
    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        return None
        
    try:
        if "sac" in MODEL_PATH.lower() or ALGORITHM.lower() == 'sac':
            model = SAC.load(MODEL_PATH, env=env)
        else:
            model = PPO.load(MODEL_PATH, env=env)
        print(f"Model loaded successfully. Model observation space: {model.observation_space.shape}")
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
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_data = {
            "timestamp": info.get("date"),
            "date": info.get("date"),
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
        buys = (actions > 0.1).sum()
        sells = (actions < -0.1).sum()
        num_trades = buys + sells
        print(f"Simulation Period: {start_timestamp} to {end_timestamp}")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Steps: {len(history)}")
        print(f"Number of Trades: {num_trades}")
        print(f"Buys: {buys}, Sells: {sells}")
        print(f"Actions: min={actions.min():.4f}, max={actions.max():.4f}, mean={actions.mean():.4f}")
    return pd.DataFrame(history)

def create_plot(df, start_timestamp=None, end_timestamp=None, include_plotlyjs=True):
    """Creates the Plotly interactive chart."""
    if start_timestamp or end_timestamp:
        df = df.copy()
        if start_timestamp:
            df = df[df['timestamp'] >= pd.to_datetime(start_timestamp)]
        if end_timestamp:
            df = df[df['timestamp'] <= pd.to_datetime(end_timestamp)]
        if df.empty:
            # Return empty plot
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                subplot_titles=("Price Action & Trades", "Net Worth", "USDT", "BTC")
            )
            return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id='chart')

    # Resample to daily to reduce data points for plotting
    df = df.set_index('timestamp')
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

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=("Price Action & Trades", "Net Worth", "USDT", "BTC")
    )

    # 1. Price
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)

    # Markers
    buys = df[df['action'] > 0.1]
    fig.add_trace(go.Scatter(
        x=buys['timestamp'], y=buys['close'], mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12), name='Buy'
    ), row=1, col=1)

    sells = df[df['action'] < -0.1]
    fig.add_trace(go.Scatter(
        x=sells['timestamp'], y=sells['close'], mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12), name='Sell'
    ), row=1, col=1)

    # 2. Net Worth
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['net_worth'], line=dict(color='#00bfff', width=2), name='Net Worth'
    ), row=2, col=1)

    # 3. USDT
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['balance_usdt'], line=dict(color='#00ff00', width=1), fill='tozeroy', name='USDT'
    ), row=3, col=1)

    # 4. BTC
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['shares_held'], line=dict(color='#ffa500', width=1), fill='tozeroy', name='BTC'
    ), row=4, col=1)

    fig.update_layout(
        title=f"AI Bot Backtest Results",
        height=1200,
        template="plotly_dark",
        hovermode="x unified",
        dragmode='pan',
        margin=dict(l=40, r=40, t=70, b=40),
        xaxis=dict(range=[df['timestamp'].min(), df['timestamp'].max()]),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(type='date')
    # Ensure autoscaling for y-axes
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

    buys = df[df['action'] > 0.1]
    buys_data = {
        'x': [d.isoformat() for d in buys['timestamp']],
        'y': buys['close'].tolist(),
        'mode': 'markers',
        'marker': {'symbol': 'triangle-up', 'color': 'green', 'size': 12},
        'name': 'Buy',
        'xaxis': 'x',
        'yaxis': 'y'
    }

    sells = df[df['action'] < -0.1]
    sells_data = {
        'x': [d.isoformat() for d in sells['timestamp']],
        'y': sells['close'].tolist(),
        'mode': 'markers',
        'marker': {'symbol': 'triangle-down', 'color': 'red', 'size': 12},
        'name': 'Sell',
        'xaxis': 'x',
        'yaxis': 'y'
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

    layout = {
        'title': 'AI Bot Backtest Results',
        'height': 1200,
        'template': 'plotly_dark',
        'hovermode': 'x unified',
        'dragmode': 'pan',
        'margin': {'l': 40, 'r': 40, 't': 70, 'b': 40},
        'paper_bgcolor': '#111111',
        'plot_bgcolor': '#111111',
        'xaxis': {'type': 'date', 'range': [timestamps_iso[0], timestamps_iso[-1]], 'domain': [0, 1], 'rangeslider': {'visible': False}},
        'yaxis': {'domain': [0.5, 1], 'autorange': True},
        'xaxis2': {'matches': 'x', 'showticklabels': False, 'domain': [0, 1]},
        'yaxis2': {'domain': [0.35, 0.5], 'autorange': True},
        'xaxis3': {'matches': 'x', 'showticklabels': False, 'domain': [0, 1]},
        'yaxis3': {'domain': [0.15, 0.35], 'autorange': True},
        'xaxis4': {'matches': 'x', 'showticklabels': False, 'domain': [0, 1]},
        'yaxis4': {'domain': [0, 0.15], 'autorange': True}
    }

    return {
        'traces': [candlestick, buys_data, sells_data, net_worth, balance, shares],
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

    # Filter from default test-split date
    start_timestamp = pd.to_datetime("2024-01-01")
    filtered_df = GLOBAL_RESULTS[GLOBAL_RESULTS['timestamp'] >= start_timestamp].copy()
    csv_data = filtered_df.to_csv(index=False)
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

    # Initially show 5 months of data
    max_timestamp = GLOBAL_RESULTS['timestamp'].max()
    global current_start, current_end
    current_end = max_timestamp
    current_start = max_timestamp - pd.DateOffset(months=5)
    initial = GLOBAL_RESULTS.iloc[0]['net_worth']
    final = GLOBAL_RESULTS.iloc[-1]['net_worth']
    roi = ((final - initial) / initial) * 100

    html_chart = create_plot(GLOBAL_RESULTS, start_timestamp=current_start, end_timestamp=current_end)

    html = render_template_string("""
        <!doctype html>
        <html>
            <head>
                <title>AI Bot Backtest</title>
                <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
                <style>
                    body { font-family: sans-serif; margin: 0; padding: 0; background: #111; color: #eee; }
                    .stats { display: flex; gap: 20px; background: #222; padding: 15px; border-bottom: 1px solid #333; justify-content: center; }
                    .stat-box { text-align: center; }
                    .val { font-size: 1.5em; font-weight: bold; }
                    .pos { color: #4caf50; }
                    .neg { color: #f44336; }
                </style>
            </head>
            <body>
                <div class="stats">
                    <div class="stat-box">
                        <div>Initial</div>
                        <div class="val">${{ "%.2f"|format(initial) }}</div>
                    </div>
                    <div class="stat-box">
                        <div>Final</div>
                        <div class="val">${{ "%.2f"|format(final) }}</div>
                    </div>
                    <div class="stat-box">
                        <div>ROI</div>
                        <div class="val {{ 'pos' if roi > 0 else 'neg' }}">{{ "%.2f"|format(roi) }}%</div>
                    </div>
                </div>
                <div class="controls" style="text-align: center; padding: 10px; background: #222; border-bottom: 1px solid #333;">
                    <button id="pan_left" style="margin: 0 10px; padding: 10px 20px; background: #444; color: #eee; border: none; cursor: pointer;">← Pan Left (1 Month)</button>
                    <button id="pan_right" style="margin: 0 10px; padding: 10px 20px; background: #444; color: #eee; border: none; cursor: pointer;">Pan Right (1 Month) →</button>
                </div>
                <div style="padding: 20px;">
                    {{ chart|safe }}
                </div>
                <script>
                    var socket = io();
                    var currentStart = null;
                    var currentEnd = null;
                    function attachListener() {
                        var plotDiv = document.getElementById('chart');
                        if (plotDiv) {
                            plotDiv.on('plotly_relayout', function(data) {
                                console.log('relayout data:', data);
                                if (data['xaxis.range[0]'] && data['xaxis.range[1]']) {
                                    var newStart = data['xaxis.range[0]'];
                                    var newEnd = data['xaxis.range[1]'];
                                    if (newStart !== currentStart || newEnd !== currentEnd) {
                                        currentStart = newStart;
                                        currentEnd = newEnd;
                                        console.log('Emitting range_change', {start: newStart, end: newEnd});
                                        socket.emit('range_change', {start: newStart, end: newEnd});
                                    }
                                }
                            });
                        }
                    }
                    attachListener();
                    document.getElementById('pan_left').addEventListener('click', function() {
                        socket.emit('pan', {direction: 'left'});
                    });
                    document.getElementById('pan_right').addEventListener('click', function() {
                        socket.emit('pan', {direction: 'right'});
                    });
                    socket.on('update_traces', function(data) {
                        console.log('Received update_traces, updating chart');
                        console.log('Received data:', data);
                        var plotDiv = document.getElementById('chart');
                        if (plotDiv) {
                            console.log('Using full layout from data');
                            Plotly.react(plotDiv, data.traces, data.layout);
                            console.log('Chart updated');
                        }
                    });
                </script>
            </body>
        </html>
    """, chart=html_chart, initial=initial, final=final, roi=roi)

    response = Response(html, mimetype='text/html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@socketio.on('range_change')
def handle_range(data):
    print(f"Received range_change: {data}")
    global current_start, current_end, GLOBAL_RESULTS
    if GLOBAL_RESULTS is None:
        try:
            with open('backtest_results.pkl', 'rb') as f:
                GLOBAL_RESULTS = pickle.load(f)
        except FileNotFoundError:
            return
    start = pd.to_datetime(data.get('start'))
    end = pd.to_datetime(data.get('end'))
    print(f"Parsed start: {start}, end: {end}")
    current_start = start
    current_end = end
    df_filtered = GLOBAL_RESULTS[(GLOBAL_RESULTS['timestamp'] >= current_start) & (GLOBAL_RESULTS['timestamp'] <= current_end)]
    print(f"Filtered df shape: {df_filtered.shape}")
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
    print(f"Daily df shape: {df.shape}")
    print("First row:", df.iloc[0].to_dict())
    print("Last row:", df.iloc[-1].to_dict())
    data_dict = get_trace_data(df)
    print("Emitting update_traces")
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
    print(f"Pan daily df shape: {df.shape}")
    print("Pan first row:", df.iloc[0].to_dict())
    print("Pan last row:", df.iloc[-1].to_dict())
    data_dict = get_trace_data(df)
    print("Emitting update_traces for pan")
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
    socketio.run(app, debug=False, use_reloader=True)
