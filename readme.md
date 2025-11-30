# 🤖 AI Trading Bot 3: Deep Reinforcement Learning for Crypto

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange) ![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green) ![WandB](https://img.shields.io/badge/Logging-WandB-yellow)

An advanced, high-performance algorithmic trading agent trained to trade **Bitcoin (BTC/USDT)** using **Soft Actor-Critic (SAC)**.

Unlike basic bots that use simple indicators, this agent "sees" the market structure through **Volume Profiles**, identifying high-value support/resistance zones (POC, VAH, VAL) combined with momentum indicators. It utilizes **Curriculum Learning** to evolve from aggressive profit-seeking to professional risk management.

---

## 🚀 Key Features

### 🧠 The Brain
*   **Algorithm:** Soft Actor-Critic (SAC) - An off-policy algorithm optimized for continuous action spaces.
*   **Network Architecture:** Deep MLP (Multi-Layer Perceptron) with `[512, 512]` neurons.
*   **Parallel Processing:** Uses `SubprocVecEnv` to run **14+ parallel traders** simultaneously on the CPU, feeding a GPU for massive throughput (1,000+ FPS).

### 👀 The Vision (Observation Space)
The bot receives a flattened vector containing:
1.  **Market Structure:** 7-Day and 30-Day **Volume Profiles** (Point of Control, Value Area High/Low) relative to current price.
2.  **Volume Heatmap:** A 100-bin normalized distribution of volume history.
3.  **Momentum:** RSI (14), Stochastic RSI (14).
4.  **Trend:** MACD Line + Signal Line.
5.  **State:** Current Portfolio Balance and Holdings.

### 🎓 The Education (Curriculum Learning)
The reward function evolves over **5,000,000 steps** to shape behavior:
*   **Phase 1 (0 - 1M Steps):** **Pure Profit.** Learn market mechanics and accumulate capital.
*   **Phase 2 (1M - 2M Steps):** **Sortino Ratio.** Penalize downside volatility. Learn to enter cleaner trades.
*   **Phase 3 (2M+ Steps):** **Drawdown & Wealth Preservation.** Heavily penalize dropping below All-Time High (ATH). Learn to protect gains.

### 🛡️ Robustness
*   **Train/Test Split:** Trains on historical data (e.g., 2017-2021) and rigorously evaluates on unseen future data (2022-2025) every 50k steps.
*   **Reality Simulation:** Includes **0.15% trading fees** and an "Inertia Penalty" to prevent spam-trading/scalping noise.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+ (or Docker)
*   NVIDIA GPU (Recommended for speed)

### Option A: Local Install (Windows/Linux)
1.  **Clone the repo:**
    ```bash
    git clone https://github.com/zhivko/ai-trading-bot-3.git
    cd ai-trading-bot-3
    ```
2.  **Create venv:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
    ```
3.  **Install Dependencies:**
    *   *Note: For RTX 30/40/50 series, install PyTorch manually first to get CUDA support.*
    ```bash
    pip install -r requirements.txt
    ```

### Option B: Docker (Recommended for RTX 5090 / Newer Hardware)
If you have driver compatibility issues, run inside NVIDIA's optimized container.

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3
docker run --gpus all -it --rm --ipc=host -v "c:\git\ai-trading-bot-3:/workspace/bot" nvcr.io/nvidia/pytorch:25.01-py3
```

## Usage
Usage example:
```bash
python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 4096
```

or with resume if you break leaarning

```bash
python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 5000000 --wandb --device cuda --resume --batch-size 2048
```
