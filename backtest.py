import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template_string
from stable_baselines3 import PPO, SAC

# Import your custom environment
from trading_env import TradingEnv

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("models", "BTCUSDT_best_eval", "best_model.zip")
DATA_PATH = os.path.join("", "BTCUSDT_data.csv") 
ALGORITHM = "PPO" 

# Global Cache
CACHED_DF = None

def load_data():
    """Loads and prepares data once."""
    global CACHED_DF, DATA_PATH
    if CACHED_DF is not None:
        return CACHED_DF

    print(f"Loading CSV Data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        # Handle fallback if file is just in root
        if os.path.exists("BTCUSDT_data.csv"):
             DATA_PATH = "BTCUSDT_data.csv"
        else:
            raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # 2. Rename timestamp -> date
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)
    
    # 3. Datetime conversion
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    CACHED_DF = df
    return df

def run_simulation():
    """Runs the model against the environment."""
    df = load_data()
    
    print("Initializing Environment...")
    # Env will now auto-load your pickled VP files
    env = TradingEnv(df, initial_balance=1000, lookback_window=50)
    
    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}"

    try:
        if "sac" in MODEL_PATH.lower() or ALGORITHM.lower() == 'sac':
            model = SAC.load(MODEL_PATH, env=env)
        else:
            model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

    print("Running Backtest Loop...")
    obs, _ = env.reset()
    done = False
    
    history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Capture step data
        step_data = {
            "date": info.get("date"), 
            "close": info.get('price'),
            "net_worth": info.get('portfolio_value'),
            "balance_usdt": info.get('balance'),
            "shares_held": info.get('shares_held'),
            "action": float(action[0]),
            
            # OHLC for chart
            "open": df.iloc[env.current_step]['open'],
            "high": df.iloc[env.current_step]['high'],
            "low": df.iloc[env.current_step]['low'],
        }
        history.append(step_data)

    return pd.DataFrame(history), None

def create_plot(df):
    """Creates the Plotly interactive chart with 4 aligned subplots."""
    
    # Create Subplots: 4 Rows, Shared X Axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=("Price Action & Trades", "Net Worth (Total Portfolio)", "USDT Balance", "BTC / Shares Held")
    )

    # --- 1. OHLC Chart (Top) ---
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Price'
    ), row=1, col=1)

    # Buy Markers
    buys = df[df['action'] > 0.1] 
    fig.add_trace(go.Scatter(
        x=buys['date'], y=buys['close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy'
    ), row=1, col=1)

    # Sell Markers
    sells = df[df['action'] < -0.1] 
    fig.add_trace(go.Scatter(
        x=sells['date'], y=sells['close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell'
    ), row=1, col=1)

    # --- 2. Net Worth ---
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['net_worth'],
        line=dict(color='#00bfff', width=2),
        name='Net Worth'
    ), row=2, col=1)

    # --- 3. USDT Balance ---
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['balance_usdt'],
        line=dict(color='#00ff00', width=1),
        fill='tozeroy',
        name='USDT Balance'
    ), row=3, col=1)

    # --- 4. BTC Balance ---
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['shares_held'],
        line=dict(color='#ffa500', width=1),
        fill='tozeroy',
        name='BTC Held'
    ), row=4, col=1)

    # --- Layout Settings ---
    fig.update_layout(
        title=f"AI Bot Backtest Results",
        xaxis_rangeslider_visible=False,
        height=1200,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Add Range Slider to the bottom chart (controls all shared axes)
    fig.update_xaxes(rangeslider=dict(visible=True), row=4, col=1)

    return fig.to_html(full_html=True)

# --- FLASK ROUTES ---

@app.route('/')
def index():
    df, error = run_simulation()
    
    if error:
        return f"<h1>Error</h1><p>{error}</p>"
    
    if df is None or df.empty:
        return "<h1>No trades made or data empty.</h1>"
        
    # Generate Stats
    initial = df.iloc[0]['net_worth']
    final = df.iloc[-1]['net_worth']
    roi = ((final - initial) / initial) * 100
    
    html_chart = create_plot(df)
    
    return render_template_string("""
        <!doctype html>
        <html>
            <head>
                <title>AI Bot Backtest</title>
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
                
                {{ chart|safe }}
            </body>
        </html>
    """, chart=html_chart, initial=initial, final=final, roi=roi)

if __name__ == "__main__":
    print("Starting Backtest Server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, use_reloader=True)