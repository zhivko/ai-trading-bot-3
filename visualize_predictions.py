# visualize_predictions.py
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from captum.attr import IntegratedGradients

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from enhanced_trading_env import EnhancedTradingEnv
from fetch_metrics import generate_metrics
import os


def compute_saliency(model, obs_tensor, lstm_states, device='cuda'):
    was_training = model.policy.training
    model.policy.eval()

    def forward_func(inputs):
        batch_size = inputs.shape[0]
        if lstm_states is not None:
            h, c = lstm_states
            h = torch.as_tensor(h, device=device).repeat(1, batch_size, 1)
            c = torch.as_tensor(c, device=device).repeat(1, batch_size, 1)
        else:
            num_layers = model.policy.lstm_actor.num_layers
            hidden_size = model.policy.lstm_actor.hidden_size
            h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        episode_starts = torch.zeros(batch_size, device=device)
        dist, _ = model.policy.get_distribution(inputs, (h, c), episode_starts)
        return dist.distribution.mean.sum(dim=1)

    ig = IntegratedGradients(forward_func)
    obs_tensor.requires_grad = True
    attr, delta = ig.attribute(obs_tensor, n_steps=30, return_convergence_delta=True)
    model.policy.train(was_training)
    return attr.detach().cpu().numpy()[0].squeeze()


def plot_trading_chart(df_results, feature_names, attributions_matrix):
    """
    Beautiful trading visualization with price, EMA, bull/bear zones, smart trade markers, and exposure.
    """
    prices = df_results['close'].values
    actions = df_results['action'].values
    ema50 = df_results['ema50'].values
    net_worths = df_results['net_worth'].values
    dates = df_results.index
    net_worth_final = df_results['net_worth'].iloc[-1]

    steps = np.arange(len(prices))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1]}
    )

    # === Price + EMA + Bull/Bear shading ===
    ax1.plot(steps, prices, label='Price', color='black', linewidth=1.4)
    ax1.plot(steps, ema50, label='EMA 50', color='orange', linestyle='--', linewidth=1.2)

    # Bull/Bear background
    ax1.fill_between(
        steps, prices, ema50,
        where=(prices >= ema50),
        color='green', alpha=0.12, interpolate=True, label='Bull Regime'
    )
    ax1.fill_between(
        steps, prices, ema50,
        where=(prices < ema50),
        color='red', alpha=0.12, interpolate=True, label='Bear Regime'
    )

    # === Smart Buy/Sell Markers (based on exposure change) ===
    buy_plotted = sell_plotted = False
    for i in range(1, len(actions)):
        delta = actions[i] - actions[i-1] if i > 0 else 0
        if abs(delta) < 0.02:  # Filter noise
            continue
        if delta > 0:  # Increasing long or reducing short → Buy signal
            label = 'Buy' if not buy_plotted else ""
            ax1.scatter(steps[i], prices[i], color='green', marker='^', s=120, zorder=5,
                        edgecolors='darkgreen', linewidth=1.5, label=label)
            buy_plotted = True
        elif delta < 0:  # Increasing short or reducing long → Sell signal
            label = 'Sell' if not sell_plotted else ""
            ax1.scatter(steps[i], prices[i], color='red', marker='v', s=120, zorder=5,
                        edgecolors='darkred', linewidth=1.5, label=label)
            sell_plotted = True

    ax1.set_title(f"AI Trading Bot Performance | Final Portfolio Value: ${net_worth_final:,.2f}", fontsize=16, pad=20)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Price (USDT)")

    # === Exposure Bar Chart ===
    colors = ['green' if a >= 0 else 'red' for a in actions]
    ax2.bar(steps, actions, color=colors, width=0.9, alpha=0.8)
    ax2.axhline(0, color='white', linewidth=1.2)
    ax2.set_ylabel("Exposure", fontsize=12)
    ax2.grid(True, axis='y', alpha=0.4)

    # === Net Worth Plot ===
    ax3.plot(steps, net_worths, label='Net Worth', color='blue', linewidth=1.5)
    ax3.set_ylabel("Net Worth (USDT)")
    ax3.set_xlabel("Time")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # === Smart Date Ticks ===
    n_ticks = 10
    indices = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    tick_labels = [dates[i].strftime("%m-%d\n%H:%M") for i in indices]
    ax3.set_xticks(indices)
    ax3.set_xticklabels(tick_labels, fontsize=10, ha='center')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/trading_performance.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("Trading performance chart saved: results/trading_performance.png")

    # === Average Saliency (Top 20) ===
    avg_importance = np.mean(np.abs(attributions_matrix), axis=0)
    top_idx = np.argsort(avg_importance)[-20:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = avg_importance[top_idx]

    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_values)), top_values, color='skyblue', edgecolor='navy', alpha=0.8)
    plt.yticks(range(len(top_values)), top_features, fontsize=10)
    plt.xlabel("Average Absolute Saliency", fontsize=12)
    plt.title("Top 20 Most Important Features (Neural Network Focus)", fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/average_saliency.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("Average saliency chart saved: results/average_saliency.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--model-path", type=str, default="models/best_model")
    parser.add_argument("--data-path", type=str, default="BTCUSDT_data.csv")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to visualize")
    parser.add_argument("--start-index", type=int, default=5000, help="Start step in dataset")
    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    if not os.path.exists(args.model_path + '.zip'):
        print(f"Model file does not exist at {args.model_path}.zip")
    print("Loading data...")
    df = pd.read_csv(args.data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    print("Setting up environment...")
    env = EnhancedTradingEnv(
        df=df,
        lookback_window=1,  # Critical for RecurrentPPO
        initial_balance=10000,
    )
    env = DummyVecEnv([lambda: env])

    # Load normalization stats if exist
    norm_path = args.model_path.replace(".zip", ".pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False

    print(f"Loading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecurrentPPO.load(args.model_path, env=env, device=device)

    print("Running inference and computing saliency...")
    obs = env.reset()
    lstm_states = None

    actions = []
    prices = []
    ema50s = []
    net_worths = []
    timestamps = []
    attributions_list = []

    # Get feature names
    try:
        feature_names = env.envs[0].get_feature_names()
    except:
        feature_names = [f"F{i}" for i in range(env.observation_space.shape[1])]

    for i in tqdm(range(args.steps)):
        # Saliency
        # FIX: obs is (Batch, Features). We need (Batch, Seq_Len, Features).
        # We insert the sequence dimension at index 1.
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(1)

        lstm_states_np = lstm_states
        attr = compute_saliency(model, obs_tensor, lstm_states_np, device)
        attributions_list.append(attr)

        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, _, done, infos = env.step(action)

        info = infos[0]
        actions.append(action[0][0])
        prices.append(info['price'])
        ema50s.append(info.get('ema50', info['price']))  # fallback
        net_worths.append(info['net_worth'])
        timestamps.append(env.envs[0].raw_df.iloc[env.envs[0].current_step].name)

        if done:
            break

    # Build results DataFrame
    results_df = pd.DataFrame({
        'close': prices,
        'action': actions,
        'net_worth': net_worths,
        'ema50': ema50s
    }, index=pd.DatetimeIndex(timestamps))

    attributions_matrix = np.stack(attributions_list)

    # Generate charts
    plot_trading_chart(results_df, feature_names, attributions_matrix)
    print("\nVisualization complete!")
    generate_metrics()


if __name__ == "__main__":
    main()