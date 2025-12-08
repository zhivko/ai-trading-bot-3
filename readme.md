# 🤖 AI Trading Bot 3: Deep Reinforcement Learning for Crypto

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange) ![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green) ![WandB](https://img.shields.io/badge/Logging-WandB-yellow)

An AI-powered trading bot using **RecurrentPPO (LSTM)** for temporal memory and advanced feature engineering.

Unlike basic bots that use simple indicators, this agent "sees" the market structure through **Volume Profiles**, identifying high-value support/resistance zones (POC, VAH, VAL) combined with momentum indicators. It utilizes **Curriculum Learning** to evolve from aggressive profit-seeking to professional risk management.

---
## 🧠 v3.0 Architecture: RecurrentPPO
The bot has been upgraded from a standard MLP policy to a Recurrent Neural Network (LSTM).

- **Algorithm:** Recurrent Proximal Policy Optimization (RecurrentPPO) via `sb3-contrib`.
- **Temporal Memory:** Uses LSTM hidden states to remember past market conditions, eliminating the need for large input sliding windows.
- **Input Shape:** `Window_Size=1`. The model sees only the current candle; history is maintained internally by the LSTM.

## 📊 Advanced Feature Engineering

### 1. Decayed Divergence Signals
Standard divergence signals (RSI/Stoch higher lows vs Price lower lows) are sparse events (single timestep). 
- **Problem:** Standard AI models miss these "blips".
- **Solution:** **Signal Decay**. When a divergence is detected, the feature spikes to `1.0` and slowly decays (e.g., `*= 0.95`) over subsequent steps. This creates a "fading memory" trace the LSTM can easily latch onto.

### 2. Volume Profile Heatmap
- Dynamically calculates the Volume Profile (Price-by-Volume) for the last 7 and 30 days.
- Normalizes this into a **40-bin Heatmap** fed directly to the agent, allowing it to "see" support and resistance zones based on liquidity.

### 3. Interpretable AI (Saliency)
- Includes a custom `RecurrentFeatureSaliencyCallback`.
- Uses **Integrated Gradients** to visualize exactly which features (Volume, RSI, Trend) triggered a specific Buy/Sell decision, accounting for the LSTM's hidden state.

## 💰 Smart Reward Function (Anti-Churn)
To prevent the AI from "churning" (rapidly flipping Buy/Sell to farm tiny fluctuations), we implemented a **Dynamic Churn Penalty**:

- **Concept:** "Healthy" trades should last a minimum duration (e.g., 24 hours).
- **Mechanism:** 
  - If a trade is closed immediately (Step 1): **Max Penalty** (-0.5 reward).
  - If a trade is held for target duration (Step 24+): **Zero Penalty**.
  - Between 0-24 steps: Penalty decays linearly.
- **Result:** The agent learns patience and only enters trades where the expected profit outweighs the "Early Exit Fee".


## 🚀 Key Features

### 🧠 The Brain
*   **Algorithm:** Recurrent Proximal Policy Optimization (RecurrentPPO) via `sb3-contrib`.
*   **Network Architecture:** Recurrent Neural Network (LSTM) for temporal memory.
*   **Input Shape:** `Window_Size=1`. The model sees only the current candle; history is maintained internally by the LSTM.
*   **Parallel Processing:** Uses `SubprocVecEnv` to run **16 parallel traders** simultaneously on the CPU, feeding a GPU for massive throughput (1,000+ FPS).

### 👀 The Vision (Observation Space)
The bot receives a flattened vector containing:
1.  **Market Structure:** 7-Day and 30-Day **Volume Profiles** (Point of Control, Value Area High/Low) relative to current price.
2.  **Volume Heatmap:** A 40-bin normalized distribution of volume history.
3.  **Decayed Divergence Signals:** Fading memory traces of RSI/Stoch divergences.
4.  **Momentum:** RSI (14), Stochastic RSI (14).
5.  **Trend:** MACD Line + Signal Line.
6.  **State:** Current Portfolio Balance and Holdings.

### 🎓 The Education (Smart Reward Function - Anti-Churn)
To prevent the AI from "churning" (rapidly flipping Buy/Sell to farm tiny fluctuations), we implemented a **Dynamic Churn Penalty**:
*   **Concept:** "Healthy" trades should last a minimum duration (e.g., 24 hours).
*   **Mechanism:**
  - If a trade is closed immediately (Step 1): **Max Penalty** (-0.5 reward).
  - If a trade is held for target duration (Step 24+): **Zero Penalty**.
  - Between 0-24 steps: Penalty decays linearly.
*   **Result:** The agent learns patience and only enters trades where the expected profit outweighs the "Early Exit Fee".

### 🛡️ Robustness
*   **Train/Test Split:** Trains on historical data (e.g., 2017-2021) and rigorously evaluates on unseen future data (2022-2025) every 50k steps.
*   **Reality Simulation:** Includes **0.15% trading fees** and an "Inertia Penalty" to prevent spam-trading/scalping noise.

---
# Train with RecurrentPPO and 16 parallel environments
python main.py --algo recurrentppo --pair BTCUSDT --n-envs 16 --window-size 1 --device cuda --wandb

## 🛠️ Installation

### Prerequisites
*   Python 3.12+ (or Docker)
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
    *  I built RTX5090 x64 drivers on win64 environment - hence --no-deps should be used (for more about build proces see below)
    ```bash
    pip install -r .\requirements.txt --no-deps
    ```

### Option B: Docker (Recommended for RTX 5090 / Newer Hardware)
If you have driver compatibility issues, run inside NVIDIA's optimized container.

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3
# first time
docker run --gpus all -it --rm --ipc=host -v "c:\git\ai-trading-bot-3:/workspace/bot" nvcr.io/nvidia/pytorch:25.01-py3
cd bot
source .venv/Scripts/activate
pip install -r requirements
```

after you save it with (run in docker host)
```bash
docker commit 802dad3739eb my-saved-pytorch-image:latest
sha256:a60aca26f88eb88e5b078dff2be2d0dda3583b28e4c98122ced32556f336c8bd
```

```python
python main.py --pair BTCUSDT --vp-days 7 30 --algo recurrentppo --test-split 2023-06-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 1024
```


... and you can start it with
```bash
docker run --gpus all -it --ipc=host -v "c:\git\ai-trading-bot-3:/workspace/bot" my-saved-pytorch-image:latest
```

## Results

EVAL REPORT (Step 4900000): ROI: 260.57% | Drawdown: 22.21%
| Section  | Metric          | Value     |
|----------|-----------------|-----------|
| rollout/ | ep_len_mean     | 1.95e+04  |
| rollout/ | ep_rew_mean     | 451       |
| time/    | episodes        | 252       |
| time/    | fps             | 1090      |
| time/    | time_elapsed    | 4520      |
| time/    | total_timesteps | 4930212   |
| train/   | actor_loss      | 4.76      |
| train/   | critic_loss     | 0.254     |
| train/   | ent_coef        | 0.00294   |
| train/   | ent_coef_loss   | 0.762     |
| train/   | learning_rate   | 0.0003    |
| train/   | n_updates       | 352150    |


![alt text](image.png)



## Usage
Usage example:


### model = sac

```bash
python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2023-01-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 4096
```

or with resume if you break learning

```bash
python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2023-01-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 4096 --resume
```


### model = recurrentppo
```bash
python main.py --pair BTCUSDT --vp-days 7 30 --algo recurrentppo --test-split 2023-06-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 1024
```


## CODE for LLM's to comment on code
Sources of project are available in:
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/main.py
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/enhanced_trading_env.py
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/callbacks/base_callbacks.py
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/feature_saliency.py

visualisation
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/callbacks/visualize_predictions.py

results
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/results/quant_report.html
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/results/visualization.png
https://raw.githubusercontent.com/zhivko/ai-trading-bot-3/refs/heads/RecurrentPPO/results/average_saliency.png



## to kILL python processes
```cmd
taskkill /F /IM python.exe
```
