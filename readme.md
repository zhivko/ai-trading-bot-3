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
docker run --gpus all -it --rm --ipc=host -v "c:\git\ai-trading-bot-3:/workspace/bot" nvcr.io/nvidia/pytorch:25.01-py3
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

## Output from logs
Below is example how to run from docker container.
```
(.venv) (base) PS C:\git\ai-trading-bot-3> python main.py --pair BTCUSDT --vp-days 7 30 --algo recurrentppo --test-split 2023-01-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 4096
Starting main function...
Parsed args: Namespace(pair='BTCUSDT', timeframe='1h', initial_balance=10000, trading_fee=0.00075, buy_threshold=0.3, sell_threshold=-0.3, vp_days=[7, 30], vp_bins=40, window_size=50, n_envs=15, phase=1, algo='recurrentppo', total_timesteps=5000000, batch_size=4096, learning_rate=0.0001, device='cuda', wandb=True, resume=False, test_split='2023-01-01', seed=42)
Random seed set.
Loading data from BTCUSDT_data.csv...
C:\git\ai-trading-bot-3\main.py:114: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df = df.fillna(method='bfill').fillna(method='ffill')
Split Data: Train (46981) | Test (25512)
Creating 15 parallel environments...
--- Initializing EnhancedTradingEnv (Target Bins: 40) ---
Calculating VP for training data...
Calculating Rolling VP for 7 days (Bins: 40)...
⚡ [VP] Loaded cached 7d profile (Bins: 40)
Calculating Rolling VP for 30 days (Bins: 40)...
⚡ [VP] Loaded cached 30d profile (Bins: 40)
Train VP calculation complete.
Calculating VP for test data...
⚡ [VP] Loaded cached 7d profile (Bins: 40)
⚡ [VP] Loaded cached 30d profile (Bins: 40)
Test VP calculation complete.
Setting up environments...
Env kwargs: {'initial_balance': 10000, 'vp_days': [7, 30], 'vp_bins': 40, 'lookback_window': 1, 'buy_threshold': 0.3, 'sell_threshold': -0.3, 'trading_fee_multiplier': 0.00075, 'phase': 1}
Creating training environment...
Training environment created.
Applying VecNormalize to training env...
VecNormalize applied.
Creating evaluation environment...
Evaluation environment created.
Setting up W&B...
wandb: Currently logged in as: zhivko. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.23.1 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.17.5
wandb: Run data is saved locally in C:\git\ai-trading-bot-3\wandb\run-20251205_150252-lvbmzzhq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run RecurrentPPO_20251205_150250_recurrentppo_BTCUSDT_VP40_Envs15
wandb:  View project at https://wandb.ai/zhivko/ai-trading-bot
wandb:  View run at https://wandb.ai/zhivko/ai-trading-bot/runs/lvbmzzhq
W&B initialized.
Initializing new RECURRENTPPO model...
Using cuda device
C:\git\ai-trading-bot-3\.venv\Lib\site-packages\torch\nn\modules\rnn.py:248: UserWarning: PyTorch was compiled without cuDNN/MIOpen support. To use cuDNN/MIOpen, rebuild PyTorch making sure the library is visible to the build system.
  or not torch.backends.cudnn.is_acceptable(fw)
C:\git\ai-trading-bot-3\.venv\Lib\site-packages\stable_baselines3\common\utils.py:168: UserWarning: get_schedule_fn() is deprecated, please use FloatSchedule() instead
  warnings.warn("get_schedule_fn() is deprecated, please use FloatSchedule() instead")
C:\git\ai-trading-bot-3\.venv\Lib\site-packages\stable_baselines3\common\utils.py:214: UserWarning: constant_fn() is deprecated, please use ConstantSchedule() instead
  warnings.warn("constant_fn() is deprecated, please use ConstantSchedule() instead")
Training started... Target: 5000000 steps
Model: RECURRENTPPO, Device: cuda
Logging to ./logs/recurrentppo_tensorboard\RecurrentPPO_1
C:\git\ai-trading-bot-3\.venv\Lib\site-packages\stable_baselines3\common\callbacks.py:418: UserWarning: Training and eval env are not of the same type<stable_baselines3.common.vec_env.vec_normalize.VecNormalize object at 0x000002290E613DA0> != <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x0000022897D67350>
  warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
   1% ━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 27,975/5,000,000  [ 0:00:08 < 0:24:14 , 3,421 it/s ]
[DEBUG Step 20000] Feature Magnitudes:
  > Volume Norm Input:   0.28843  (Should be 0.0 - 1.0)
  > Trend EMA Input:     -0.00003  (Now z-scored and clipped -1 to 1)
  > Close Pct Input:     0.00159
  > RSI Norm Input:      0.49933
  > Stoch RSI Input:     0.73576
  > MACD Norm Input:     -0.42149
  > MACD Sig Norm Input: -0.43826
  > ATR Norm Input:      0.01053
  > Regime Input:        -0.00309 (-2 to 2)
  > VP Heatmap Max:      1.00 (Now normalized by sum, max <=1.0)
  > VP Heatmap Values:   [0.4352407  0.         0.         0.         0.         0.
 0.0527168  0.49055349 0.23850683 0.37343214 0.72347481 1.
 0.86299576 0.48808602 0.60868419 0.67247243 0.11592202 0.59585825
 0.74616682 0.15799411 0.43852809 0.38062626 0.23702796 0.09863186
 0.4871279  0.82953615 0.78868592 0.32383444 0.73370109 0.71710816
 0.51409352 0.28612297 0.0513162  0.0952106  0.11397692 0.32783764
 0.24268881 0.86147501 0.         0.16202192]
  > Bull Div Stoch9:    0.00000
  > Bear Div Stoch9:    0.00000
  > Bull Div Stoch14:   1.00000
  > Bear Div Stoch14:   0.02365
  > Bull Div RSI:       1.00000
  > Bear Div RSI:       0.00235
   1% ━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57,915/5,000,000  [ 0:00:14 < 0:20:20 , 4,054 it/s ]
[DEBUG Step 40000] Feature Magnitudes:
  > Volume Norm Input:   0.60769  (Should be 0.0 - 1.0)
  > Trend EMA Input:     0.00000  (Now z-scored and clipped -1 to 1)
  > Close Pct Input:     -0.04175
  > RSI Norm Input:      0.45986
  > Stoch RSI Input:     0.00000
  > MACD Norm Input:     0.57676
  > MACD Sig Norm Input: 0.45350
  > ATR Norm Input:      0.01686
  > Regime Input:        0.00011 (-2 to 2)
  > VP Heatmap Max:      1.00 (Now normalized by sum, max <=1.0)
  > VP Heatmap Values:   [0.17379429 0.04260727 0.08747734 0.02495642 0.06858435 0.16538144
 0.18396407 0.26148729 0.57221314 0.75291814 0.58371784 1.
 0.75353071 0.4022577  0.25216325 0.25518617 0.15053012 0.
 0.26898599 0.         0.         0.         0.         0.
 0.07477539 0.06598062 0.41386315 0.         0.         0.1896948
 0.         0.06917012 0.16121706 0.04708066 0.27683628 0.14089407
 0.16650838 0.22370683 0.03355613 0.07307414]
  > Bull Div Stoch9:    0.00435
  > Bear Div Stoch9:    0.03752
  > Bull Div Stoch14:   0.05954
  > Bear Div Stoch14:   0.35849
  > Bull Div RSI:       0.03056
  > Bear Div RSI:       0.03752
   1% ━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58,410/5,000,000  [ 0:00:14 < 0:20:18 , 4,058 it/s ]
[DEBUG Step 20000] Feature Magnitudes:
  > Volume Norm Input:   0.28843  (Should be 0.0 - 1.0)
  > Trend EMA Input:     -0.00003  (Now z-scored and clipped -1 to 1)
  > Close Pct Input:     0.00159
  > RSI Norm Input:      0.49933
  > Stoch RSI Input:     0.73576
  > MACD Norm Input:     -0.42149
  > MACD Sig Norm Input: -0.43826
  > ATR Norm Input:      0.01053
  > Regime Input:        -0.00309 (-2 to 2)
  > VP Heatmap Max:      1.00 (Now normalized by sum, max <=1.0)
  > VP Heatmap Values:   [0.4352407  0.         0.         0.         0.         0.
 0.0527168  0.49055349 0.23850683 0.37343214 0.72347481 1.
 0.86299576 0.48808602 0.60868419 0.67247243 0.11592202 0.59585825
 0.74616682 0.15799411 0.43852809 0.38062626 0.23702796 0.09863186
 0.4871279  0.82953615 0.78868592 0.32383444 0.73370109 0.71710816
 0.51409352 0.28612297 0.0513162  0.0952106  0.11397692 0.32783764
 0.24268881 0.86147501 0.         0.16202192]
  > Bull Div Stoch9:    0.00000
  > Bear Div Stoch9:    0.00000
  > Bull Div Stoch14:   1.00000
  > Bear Div Stoch14:   0.02365
  > Bull Div RSI:       1.00000
  > Bear Div RSI:       0.00235

[DEBUG Step 30000] Feature Magnitudes:
  > Volume Norm Input:   0.14655  (Should be 0.0 - 1.0)
  > Trend EMA Input:     0.00000  (Now z-scored and clipped -1 to 1)
  > Close Pct Input:     -0.01366
  > RSI Norm Input:      0.55680
  > Stoch RSI Input:     0.16524
  > MACD Norm Input:     0.95272
  > MACD Norm Input:     0.95272
  > MACD Sig Norm Input: 0.29701
  > ATR Norm Input:      0.01980
  > Regime Input:        0.00004 (-2 to 2)
  > VP Heatmap Max:      1.00 (Now normalized by sum, max <=1.0)
  > VP Heatmap Values:   [0.31878431 0.         0.34042255 0.         0.         0.15836506
 0.59035039 0.15254533 0.2878754  0.66256623 0.18789508 0.28647602
 0.11640599 0.36160474 0.52963375 0.         0.34848829 0.
 0.30132313 0.16220579 0.         0.         0.17830266 0.06269229
 0.29240021 1.         0.36926286 0.39166301 0.4489721  0.26609916
 0.4272737  0.5185975  0.81603432 0.65626844 0.69667649 0.48847344
 0.47621692 0.68661087 0.53376679 0.05620091]
  > Bull Div Stoch9:    0.16608
  > Bear Div Stoch9:    0.01830
  > Bull Div Stoch14:   0.16608
  > Bear Div Stoch14:   0.02134
  > Bull Div RSI:       0.46329
  > Bear Div RSI:       0.01830
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 2.63e+03 |
|    ep_rew_mean     | -670     |
| time/              |          |
|    fps             | 4023     |
|    iterations      | 1        |
|    time_elapsed    | 15       |
|    total_timesteps | 61440    |
---------------------------------
   1% ━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   2% ━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 95,160/5,000,000  [ 0:08:41 < 0:31:25 , 2,602 it/s ]
```


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
https://github.com/zhivko/ai-trading-bot-3/blob/feat/sane-action-reward-v3/main.py
https://github.com/zhivko/ai-trading-bot-3/blob/feat/sane-action-reward-v3/enhanced_trading_env.py
https://github.com/zhivko/ai-trading-bot-3/blob/feat/sane-action-reward-v3/callbacks/base_callbacks
https://github.com/zhivko/ai-trading-bot-3/blob/feat/sane-action-reward-v3/callbacks/feature_saliency.py
