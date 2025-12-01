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

## Build pytorch on windows

1. Open Visual Studio Installer
Click Windows Start -> Type "Visual Studio Installer".
Click Modify next to Visual Studio 2022.
Go to the "Individual Components" tab at the top.
Search for "Windows 11 SDK" (e.g., Windows 11 SDK (10.0.22xxx)).
Check the box and click Modify/Install.
2. Restart Developer PowerShell
After the SDK installs:
Close your current terminal.
Open Developer PowerShell for VS 2022 (Admin). In win search enter "Open Developer PowerShell for VS 2022" and rightclick - run as admin.
Run rc /? to verify it works. If it prints help text, you are good.
3. Clean and Retry
code
Powershell
conda activate gpu_env
cd C:\git\pytorch
Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue

# Reset variables
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:USE_NINJA = "1"
$env:MAX_JOBS = "8"
$env:CMAKE_GENERATOR = "Ninja"

python setup.py install

## Output from logs:

´´´
root@c00ed23b4b46:/workspace/bot# python main.py --pair BTCUSDT --vp-days 7 30 --algo sac --test-split 2022-01-01 --total-timesteps 5000000 --wandb --device cuda --batch-size 4096

🧹 FRESH START DETECTED
   - Deleting old models in ./models/BTCUSDT...
✅ Cleanup complete. Starting fresh.
Loading: BTCUSDT_data.csv
✂️ Splitting data at 2022-01-01...
🚀 Speeding up with 14 Parallel Environments...
🔥 Warming up cache on Main Process...
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
✅ Cache ready. Launching Swarm.
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_d73ac5d92319be4a4e40ee2fd2f37bca.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_55658296082812756b7a010f89ef3274.pkl
  [Cache] Loading... (this is fast)
DEBUG: raw_df columns after reset_index: ['date', 'open', 'high', 'low', 'close', 'volume']
DEBUG: Attempting to set index on column: 'date', available columns: ['date', 'open', 'high', 'low', 'close', 'volume']
--- Initializing Environment (VP Days: [7, 30]) ---
  [Cache] Found existing Volume Profile: vp_cache/vp_win7_976ae9ac9f3fb76ee8cf42c9211dd8ac.pkl
  [Cache] Loading... (this is fast)
  [Cache] Found existing Volume Profile: vp_cache/vp_win30_d171af62508219e5c2ea9a1bd85e837e.pkl
  [Cache] Loading... (this is fast)
✨ Creating NEW SAC Model on CUDA
Using cuda device
wandb: Currently logged in as: zhivko. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.23.0 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.17.5
wandb: Run data is saved locally in /workspace/bot/wandb/run-20251130_194519-uq8nauli
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run wise-feather-76
wandb: ⭐️ View project at https://wandb.ai/zhivko/ai-trading-bot
wandb: 🚀 View run at https://wandb.ai/zhivko/ai-trading-bot/runs/uq8nauli
--- STARTING TRAINING (5000000 steps) ---
Logging to ./sac_tb/SAC_1
Step 20600 [2019-12-27 12:00:00]: Price=7226 | POC=7206 (+0.29%) | NetWorth=10000 | MaxNetWorth=10000
Step 20500 [2019-12-23 08:00:00]: Price=7524 | POC=7156 (+5.14%) | NetWorth=10000 | MaxNetWorth=10000
Step 34400 [2021-07-25 13:00:00]: Price=34179 | POC=32242 (+6.01%) | NetWorth=9941 | MaxNetWorth=10000
Step 25600 [2020-07-23 08:00:00]: Price=9516 | POC=9163 (+3.85%) | NetWorth=10088 | MaxNetWorth=10093
Step 17900 [2019-09-05 20:00:00]: Price=10507 | POC=10540 (-0.32%) | NetWorth=9966 | MaxNetWorth=10000
Step 25000 [2020-06-28 08:00:00]: Price=9054 | POC=9275 (-2.38%) | NetWorth=9834 | MaxNetWorth=10004
Step 28800 [2020-12-03 17:00:00]: Price=19299 | POC=19119 (+0.94%) | NetWorth=10200 | MaxNetWorth=10239
Step 26600 [2020-09-03 00:00:00]: Price=11410 | POC=11386 (+0.22%) | NetWorth=10055 | MaxNetWorth=10105
Step 32000 [2021-04-16 08:00:00]: Price=61192 | POC=63030 (-2.92%) | NetWorth=9929 | MaxNetWorth=10044
Step 2100 [2017-11-12 22:00:00]: Price=5660 | POC=7142 (-20.75%) | NetWorth=9282 | MaxNetWorth=10240
Step 16600 [2019-07-13 08:00:00]: Price=11510 | POC=11392 (+1.03%) | NetWorth=10256 | MaxNetWorth=10352
Step 15500 [2019-05-28 12:00:00]: Price=8695 | POC=7973 (+9.06%) | NetWorth=9851 | MaxNetWorth=10159
Step 33600 [2021-06-22 05:00:00]: Price=32578 | POC=35609 (-8.51%) | NetWorth=9607 | MaxNetWorth=10140
Step 34600 [2021-08-02 21:00:00]: Price=39202 | POC=39641 (-1.11%) | NetWorth=9573 | MaxNetWorth=10087
Step 20700 [2019-12-31 16:00:00]: Price=7211 | POC=7241 (-0.40%) | NetWorth=9941 | MaxNetWorth=10159
Step 20600 [2019-12-27 12:00:00]: Price=7226 | POC=7206 (+0.29%) | NetWorth=9637 | MaxNetWorth=10011
Step 34500 [2021-07-29 17:00:00]: Price=39606 | POC=39844 (-0.60%) | NetWorth=10439 | MaxNetWorth=10593
Step 25700 [2020-07-27 12:00:00]: Price=10292 | POC=9360 (+9.96%) | NetWorth=10555 | MaxNetWorth=10557
Step 18000 [2019-09-10 00:00:00]: Price=10336 | POC=10567 (-2.18%) | NetWorth=9857 | MaxNetWorth=10085
Step 25100 [2020-07-02 12:00:00]: Price=9243 | POC=9174 (+0.75%) | NetWorth=10008 | MaxNetWorth=10042
Step 28900 [2020-12-07 21:00:00]: Price=19075 | POC=19090 (-0.08%) | NetWorth=10557 | MaxNetWorth=10558
Step 26700 [2020-09-07 04:00:00]: Price=10171 | POC=10235 (-0.63%) | NetWorth=9085 | MaxNetWorth=10105
Step 32100 [2021-04-20 14:00:00]: Price=55336 | POC=62944 (-12.09%) | NetWorth=9754 | MaxNetWorth=10104
Step 2200 [2017-11-17 02:00:00]: Price=7879 | POC=6307 (+24.93%) | NetWorth=11238 | MaxNetWorth=11289
Step 16700 [2019-07-17 12:00:00]: Price=9450 | POC=11584 (-18.42%) | NetWorth=9306 | MaxNetWorth=10427
Step 15600 [2019-06-01 16:00:00]: Price=8555 | POC=8700 (-1.67%) | NetWorth=9688 | MaxNetWorth=10159
Step 33700 [2021-06-26 09:00:00]: Price=31053 | POC=32802 (-5.33%) | NetWorth=9725 | MaxNetWorth=10784
Step 34700 [2021-08-07 01:00:00]: Price=43170 | POC=38241 (+12.89%) | NetWorth=9820 | MaxNetWorth=10087
Step 20800 [2020-01-04 20:00:00]: Price=7341 | POC=7321 (+0.27%) | NetWorth=9870 | MaxNetWorth=10159
Step 20700 [2019-12-31 16:00:00]: Price=7211 | POC=7241 (-0.40%) | NetWorth=9662 | MaxNetWorth=10011
Step 34600 [2021-08-02 21:00:00]: Price=39202 | POC=39641 (-1.11%) | NetWorth=10847 | MaxNetWorth=11255
Step 25800 [2020-07-31 16:00:00]: Price=11352 | POC=11003 (+3.17%) | NetWorth=11217 | MaxNetWorth=11217
Step 18100 [2019-09-14 04:00:00]: Price=10279 | POC=10315 (-0.35%) | NetWorth=9904 | MaxNetWorth=10085
Step 25200 [2020-07-06 16:00:00]: Price=9316 | POC=9098 (+2.40%) | NetWorth=10031 | MaxNetWorth=10042
Step 29000 [2020-12-12 01:00:00]: Price=18284 | POC=19191 (-4.73%) | NetWorth=10610 | MaxNetWorth=10657
Step 26800 [2020-09-11 08:00:00]: Price=10294 | POC=10260 (+0.33%) | NetWorth=8895 | MaxNetWorth=10105
Step 32200 [2021-04-24 18:00:00]: Price=50562 | POC=55227 (-8.45%) | NetWorth=9191 | MaxNetWorth=10104
Step 2300 [2017-11-21 06:00:00]: Price=8074 | POC=7756 (+4.10%) | NetWorth=10701 | MaxNetWorth=11289
Step 16800 [2019-07-21 16:00:00]: Price=10354 | POC=10547 (-1.83%) | NetWorth=9536 | MaxNetWorth=10427
Step 15700 [2019-06-05 20:00:00]: Price=7721 | POC=8536 (-9.55%) | NetWorth=9145 | MaxNetWorth=10159
Step 33800 [2021-06-30 13:00:00]: Price=34735 | POC=34573 (+0.47%) | NetWorth=10408 | MaxNetWorth=10784
Step 34800 [2021-08-11 05:00:00]: Price=45800 | POC=45631 (+0.37%) | NetWorth=10673 | MaxNetWorth=10754
Step 20900 [2020-01-09 00:00:00]: Price=7957 | POC=7312 (+8.82%) | NetWorth=10456 | MaxNetWorth=10618
Step 20800 [2020-01-04 20:00:00]: Price=7341 | POC=7321 (+0.27%) | NetWorth=9857 | MaxNetWorth=10011
Step 34700 [2021-08-07 01:00:00]: Price=43170 | POC=38241 (+12.89%) | NetWorth=11991 | MaxNetWorth=12007
Step 25900 [2020-08-04 20:00:00]: Price=11212 | POC=11226 (-0.13%) | NetWorth=11325 | MaxNetWorth=11458
Step 18200 [2019-09-18 08:00:00]: Price=10163 | POC=10305 (-1.38%) | NetWorth=9875 | MaxNetWorth=10085
Step 25300 [2020-07-10 20:00:00]: Price=9232 | POC=9237 (-0.06%) | NetWorth=9920 | MaxNetWorth=10171
Step 29100 [2020-12-16 05:00:00]: Price=19374 | POC=19197 (+0.92%) | NetWorth=10888 | MaxNetWorth=11006
Step 26900 [2020-09-15 12:00:00]: Price=10908 | POC=10342 (+5.47%) | NetWorth=9205 | MaxNetWorth=10105
Step 32300 [2021-04-29 01:00:00]: Price=54568 | POC=49643 (+9.92%) | NetWorth=9452 | MaxNetWorth=10104
Step 2400 [2017-11-25 10:00:00]: Price=8402 | POC=8184 (+2.66%) | NetWorth=10654 | MaxNetWorth=11289
Step 16900 [2019-07-25 20:00:00]: Price=9873 | POC=10529 (-6.23%) | NetWorth=8690 | MaxNetWorth=10427
Step 15800 [2019-06-10 00:00:00]: Price=7599 | POC=7790 (-2.45%) | NetWorth=8911 | MaxNetWorth=10159
Step 33900 [2021-07-04 17:00:00]: Price=35558 | POC=34684 (+2.52%) | NetWorth=10329 | MaxNetWorth=10784
Step 34900 [2021-08-15 13:00:00]: Price=46050 | POC=46075 (-0.05%) | NetWorth=10831 | MaxNetWorth=10947
Step 21000 [2020-01-13 04:00:00]: Price=8118 | POC=8082 (+0.45%) | NetWorth=10394 | MaxNetWorth=10618
Step 20900 [2020-01-09 00:00:00]: Price=7957 | POC=7312 (+8.82%) | NetWorth=11017 | MaxNetWorth=11144
Step 34800 [2021-08-11 05:00:00]: Price=45800 | POC=45631 (+0.37%) | NetWorth=12442 | MaxNetWorth=12501
Step 26000 [2020-08-09 00:00:00]: Price=11753 | POC=11226 (+4.69%) | NetWorth=11791 | MaxNetWorth=11794
Step 18300 [2019-09-22 12:00:00]: Price=10033 | POC=10178 (-1.42%) | NetWorth=9598 | MaxNetWorth=10085
Step 25400 [2020-07-15 00:00:00]: Price=9261 | POC=9276 (-0.16%) | NetWorth=9987 | MaxNetWorth=10171
Step 29200 [2020-12-20 09:00:00]: Price=23593 | POC=19241 (+22.62%) | NetWorth=11250 | MaxNetWorth=11370
Step 27000 [2020-09-19 16:00:00]: Price=11059 | POC=10922 (+1.26%) | NetWorth=9151 | MaxNetWorth=10105
Step 32400 [2021-05-03 05:00:00]: Price=57939 | POC=54399 (+6.51%) | NetWorth=9746 | MaxNetWorth=10104
Step 2500 [2017-11-29 14:00:00]: Price=11058 | POC=8161 (+35.50%) | NetWorth=12139 | MaxNetWorth=12142
Step 17000 [2019-07-30 00:00:00]: Price=9533 | POC=9482 (+0.53%) | NetWorth=8737 | MaxNetWorth=10427
Step 15900 [2019-06-14 04:00:00]: Price=8233 | POC=7922 (+3.93%) | NetWorth=9491 | MaxNetWorth=10159
Step 34000 [2021-07-08 21:00:00]: Price=32700 | POC=34714 (-5.80%) | NetWorth=9989 | MaxNetWorth=10784
Step 35000 [2021-08-19 17:00:00]: Price=45730 | POC=46183 (-0.98%) | NetWorth=11145 | MaxNetWorth=11228
Step 21100 [2020-01-17 08:00:00]: Price=8929 | POC=8110 (+10.10%) | NetWorth=11143 | MaxNetWorth=11207
Step 21000 [2020-01-13 04:00:00]: Price=8118 | POC=8082 (+0.45%) | NetWorth=10844 | MaxNetWorth=11144
Step 34900 [2021-08-15 13:00:00]: Price=46050 | POC=46075 (-0.05%) | NetWorth=12685 | MaxNetWorth=12935
Step 26100 [2020-08-13 04:00:00]: Price=11548 | POC=11734 (-1.58%) | NetWorth=11554 | MaxNetWorth=11876
Step 18400 [2019-09-26 16:00:00]: Price=7955 | POC=8412 (-5.44%) | NetWorth=8255 | MaxNetWorth=10085
Step 25500 [2020-07-19 04:00:00]: Price=9153 | POC=9208 (-0.60%) | NetWorth=9988 | MaxNetWorth=10171
Step 29300 [2020-12-24 17:00:00]: Price=23377 | POC=23486 (-0.47%) | NetWorth=11262 | MaxNetWorth=11370
Step 27100 [2020-09-23 20:00:00]: Price=10230 | POC=10925 (-6.36%) | NetWorth=8825 | MaxNetWorth=10105
Step 32500 [2021-05-07 09:00:00]: Price=56231 | POC=56956 (-1.27%) | NetWorth=9589 | MaxNetWorth=10104
Step 2600 [2017-12-03 18:00:00]: Price=11711 | POC=9868 (+18.68%) | NetWorth=12537 | MaxNetWorth=12537
Step 17100 [2019-08-03 04:00:00]: Price=10846 | POC=9540 (+13.69%) | NetWorth=9267 | MaxNetWorth=10427
Step 16000 [2019-06-18 08:00:00]: Price=9143 | POC=9170 (-0.30%) | NetWorth=9620 | MaxNetWorth=10159
Step 34100 [2021-07-13 01:00:00]: Price=33157 | POC=33841 (-2.02%) | NetWorth=9934 | MaxNetWorth=10784
Step 35100 [2021-08-23 21:00:00]: Price=49560 | POC=49003 (+1.14%) | NetWorth=11146 | MaxNetWorth=11325
Step 21200 [2020-01-21 12:00:00]: Price=8661 | POC=8665 (-0.04%) | NetWorth=10902 | MaxNetWorth=11325
Step 21100 [2020-01-17 08:00:00]: Price=8929 | POC=8110 (+10.10%) | NetWorth=10867 | MaxNetWorth=11144
Step 35000 [2021-08-19 17:00:00]: Price=45730 | POC=46183 (-0.98%) | NetWorth=12284 | MaxNetWorth=13035
Step 26200 [2020-08-17 08:00:00]: Price=11869 | POC=11848 (+0.18%) | NetWorth=11376 | MaxNetWorth=11876
Step 18500 [2019-09-30 20:00:00]: Price=8215 | POC=8039 (+2.18%) | NetWorth=8199 | MaxNetWorth=10085
Step 25600 [2020-07-23 08:00:00]: Price=9516 | POC=9163 (+3.85%) | NetWorth=10046 | MaxNetWorth=10171
Step 29400 [2020-12-28 22:00:00]: Price=26892 | POC=23559 (+14.14%) | NetWorth=12090 | MaxNetWorth=12187
Step 27200 [2020-09-28 00:00:00]: Price=10888 | POC=10691 (+1.84%) | NetWorth=9143 | MaxNetWorth=10105
Step 32600 [2021-05-11 13:00:00]: Price=55750 | POC=57536 (-3.10%) | NetWorth=9854 | MaxNetWorth=10169
Step 2700 [2017-12-07 22:00:00]: Price=16096 | POC=11692 (+37.67%) | NetWorth=15058 | MaxNetWorth=15058
Step 17200 [2019-08-07 08:00:00]: Price=11575 | POC=11731 (-1.33%) | NetWorth=9527 | MaxNetWorth=10427
Step 16100 [2019-06-22 12:00:00]: Price=11090 | POC=9143 (+21.29%) | NetWorth=10853 | MaxNetWorth=10938
Step 34200 [2021-07-17 05:00:00]: Price=31487 | POC=31759 (-0.86%) | NetWorth=9335 | MaxNetWorth=10784
Step 35200 [2021-08-28 01:00:00]: Price=49142 | POC=49032 (+0.22%) | NetWorth=11194 | MaxNetWorth=11325
Step 21300 [2020-01-25 16:00:00]: Price=8350 | POC=8652 (-3.50%) | NetWorth=10676 | MaxNetWorth=11325
Step 21200 [2020-01-21 12:00:00]: Price=8661 | POC=8665 (-0.04%) | NetWorth=11043 | MaxNetWorth=11144
Step 35100 [2021-08-23 21:00:00]: Price=49560 | POC=49003 (+1.14%) | NetWorth=13321 | MaxNetWorth=13348
Step 26300 [2020-08-21 12:00:00]: Price=11760 | POC=11831 (-0.60%) | NetWorth=11210 | MaxNetWorth=11876
Step 18600 [2019-10-05 00:00:00]: Price=8129 | POC=8217 (-1.06%) | NetWorth=8208 | MaxNetWorth=10085
Step 25700 [2020-07-27 12:00:00]: Price=10292 | POC=9360 (+9.96%) | NetWorth=10406 | MaxNetWorth=10406
Step 29500 [2021-01-02 02:00:00]: Price=29324 | POC=26797 (+9.43%) | NetWorth=12299 | MaxNetWorth=12385
Step 27300 [2020-10-02 04:00:00]: Price=10590 | POC=10731 (-1.31%) | NetWorth=9282 | MaxNetWorth=10105
Step 32700 [2021-05-15 17:00:00]: Price=47801 | POC=50352 (-5.07%) | NetWorth=9510 | MaxNetWorth=10186
Step 2800 [2017-12-12 02:00:00]: Price=16600 | POC=15318 (+8.37%) | NetWorth=14101 | MaxNetWorth=15373
Step 17300 [2019-08-11 12:00:00]: Price=11459 | POC=11750 (-2.48%) | NetWorth=9726 | MaxNetWorth=10427
Step 16200 [2019-06-26 16:00:00]: Price=13715 | POC=10744 (+27.65%) | NetWorth=12731 | MaxNetWorth=12731
Step 34300 [2021-07-21 09:00:00]: Price=31314 | POC=31698 (-1.21%) | NetWorth=8989 | MaxNetWorth=10784
Step 35300 [2021-09-01 05:00:00]: Price=47424 | POC=47089 (+0.71%) | NetWorth=10954 | MaxNetWorth=11325
Step 21400 [2020-01-29 20:00:00]: Price=9386 | POC=8348 (+12.43%) | NetWorth=11452 | MaxNetWorth=11454
Step 21300 [2020-01-25 16:00:00]: Price=8350 | POC=8652 (-3.50%) | NetWorth=10795 | MaxNetWorth=11144
Step 35200 [2021-08-28 01:00:00]: Price=49142 | POC=49032 (+0.22%) | NetWorth=12937 | MaxNetWorth=13475
Step 26400 [2020-08-25 16:00:00]: Price=11326 | POC=11767 (-3.75%) | NetWorth=10917 | MaxNetWorth=11876
Step 18700 [2019-10-09 04:00:00]: Price=8130 | POC=8186 (-0.69%) | NetWorth=7975 | MaxNetWorth=10085
Step 25800 [2020-07-31 16:00:00]: Price=11352 | POC=11003 (+3.17%) | NetWorth=10654 | MaxNetWorth=10824
Step 29600 [2021-01-06 06:00:00]: Price=35126 | POC=28976 (+21.22%) | NetWorth=12532 | MaxNetWorth=13212
Step 27400 [2020-10-06 08:00:00]: Price=10742 | POC=10720 (+0.21%) | NetWorth=9421 | MaxNetWorth=10105
Step 32800 [2021-05-19 21:00:00]: Price=38772 | POC=48963 (-20.81%) | NetWorth=8868 | MaxNetWorth=10186
Step 2900 [2017-12-16 06:00:00]: Price=17626 | POC=16201 (+8.80%) | NetWorth=14128 | MaxNetWorth=15373
Step 17400 [2019-08-16 00:00:00]: Price=10302 | POC=11367 (-9.36%) | NetWorth=9756 | MaxNetWorth=10427
Step 16300 [2019-06-30 20:00:00]: Price=11475 | POC=11964 (-4.09%) | NetWorth=10664 | MaxNetWorth=12899
Step 34400 [2021-07-25 13:00:00]: Price=34179 | POC=32242 (+6.01%) | NetWorth=9407 | MaxNetWorth=10784
Step 35400 [2021-09-05 09:00:00]: Price=50125 | POC=49803 (+0.65%) | NetWorth=11372 | MaxNetWorth=11462
Step 21500 [2020-02-03 00:00:00]: Price=9411 | POC=9373 (+0.41%) | NetWorth=11579 | MaxNetWorth=11737
Step 21400 [2020-01-29 20:00:00]: Price=9386 | POC=8348 (+12.43%) | NetWorth=11237 | MaxNetWorth=11275
Step 35300 [2021-09-01 05:00:00]: Price=47424 | POC=47089 (+0.71%) | NetWorth=13269 | MaxNetWorth=13475
Step 26500 [2020-08-29 20:00:00]: Price=11516 | POC=11376 (+1.23%) | NetWorth=11053 | MaxNetWorth=11876
Step 18800 [2019-10-13 08:00:00]: Price=8342 | POC=8196 (+1.78%) | NetWorth=8111 | MaxNetWorth=10085
Step 25900 [2020-08-04 20:00:00]: Price=11212 | POC=11226 (-0.13%) | NetWorth=10218 | MaxNetWorth=10972
Step 29700 [2021-01-10 10:00:00]: Price=40429 | POC=40657 (-0.56%) | NetWorth=12954 | MaxNetWorth=13212
Step 27500 [2020-10-10 12:00:00]: Price=11340 | POC=10600 (+6.98%) | NetWorth=9932 | MaxNetWorth=10105
Step 32900 [2021-05-24 01:00:00]: Price=35089 | POC=37296 (-5.92%) | NetWorth=10410 | MaxNetWorth=10440
Step 3000 [2017-12-20 10:00:00]: Price=17242 | POC=18624 (-7.42%) | NetWorth=15765 | MaxNetWorth=15765
Step 17500 [2019-08-20 04:00:00]: Price=10811 | POC=10360 (+4.36%) | NetWorth=10466 | MaxNetWorth=10489
Step 16400 [2019-07-05 00:00:00]: Price=11171 | POC=11753 (-4.96%) | NetWorth=10742 | MaxNetWorth=12899
Step 34500 [2021-07-29 17:00:00]: Price=39606 | POC=39844 (-0.60%) | NetWorth=10148 | MaxNetWorth=10784
Step 35500 [2021-09-09 13:00:00]: Price=46876 | POC=50145 (-6.52%) | NetWorth=10908 | MaxNetWorth=11770
Step 21600 [2020-02-07 04:00:00]: Price=9792 | POC=9354 (+4.69%) | NetWorth=11878 | MaxNetWorth=11916
Step 21500 [2020-02-03 00:00:00]: Price=9411 | POC=9373 (+0.41%) | NetWorth=11541 | MaxNetWorth=11541
Step 35400 [2021-09-05 09:00:00]: Price=50125 | POC=49803 (+0.65%) | NetWorth=13349 | MaxNetWorth=13529
Step 26600 [2020-09-03 00:00:00]: Price=11410 | POC=11386 (+0.22%) | NetWorth=11198 | MaxNetWorth=11876
Step 18900 [2019-10-17 12:00:00]: Price=8046 | POC=8314 (-3.21%) | NetWorth=7845 | MaxNetWorth=10085
Step 26000 [2020-08-09 00:00:00]: Price=11753 | POC=11226 (+4.69%) | NetWorth=10506 | MaxNetWorth=10972
...
2025-12-01 01:06:07
Step 33800 [2025-11-09 09:00:00]: P=101657 | POC=101835 (-0.17%) | Port=40824 | ATH=44480
2025-12-01 01:06:07
Step 33900 [2025-11-13 13:00:00]: P=102326 | POC=102104 (+0.22%) | Port=41128 | ATH=44480
2025-12-01 01:06:07
Step 34000 [2025-11-17 17:00:00]: P=92767 | POC=95762 (-3.13%) | Port=37501 | ATH=44480
2025-12-01 01:06:07
Step 34100 [2025-11-21 21:00:00]: P=85182 | POC=91469 (-6.87%) | Port=35849 | ATH=44480
2025-12-01 01:06:07
Step 34200 [2025-11-26 01:00:00]: P=87922 | POC=87333 (+0.67%) | Port=35696 | ATH=44480
2025-12-01 01:06:08
Eval num_timesteps=4900000, episode_reward=16.94 +/- 0.00
2025-12-01 01:06:08
Episode length: 29187.00 +/- 0.00
2025-12-01 01:06:08
---------------------------------
2025-12-01 01:06:08
| eval/              |          |
2025-12-01 01:06:08
|    mean_ep_length  | 2.92e+04 |
2025-12-01 01:06:08
|    mean_reward     | 16.9     |
2025-12-01 01:06:08
| time/              |          |
2025-12-01 01:06:08
|    total_timesteps | 4900000  |
2025-12-01 01:06:08
| train/             |          |
2025-12-01 01:06:08
|    actor_loss      | 4.7      |
2025-12-01 01:06:08
|    critic_loss     | 0.303    |
2025-12-01 01:06:08
|    ent_coef        | 0.00264  |
2025-12-01 01:06:08
|    ent_coef_loss   | -0.623   |
2025-12-01 01:06:08
|    learning_rate   | 0.0003   |
2025-12-01 01:06:08
|    n_updates       | 349992   |
2025-12-01 01:06:08
---------------------------------
2025-12-01 01:06:08

2025-12-01 01:06:08
📊 EVAL REPORT (Step 4900000): ROI: 260.57% | Drawdown: 22.21%
2025-12-01 01:06:34
---------------------------------
2025-12-01 01:06:34
| rollout/           |          |
2025-12-01 01:06:34
|    ep_len_mean     | 1.95e+04 |
2025-12-01 01:06:34
|    ep_rew_mean     | 451      |
2025-12-01 01:06:34
| time/              |          |
2025-12-01 01:06:34
|    episodes        | 252      |
2025-12-01 01:06:34
|    fps             | 1090     |
2025-12-01 01:06:34
|    time_elapsed    | 4520     |
2025-12-01 01:06:34
|    total_timesteps | 4930212  |
2025-12-01 01:06:34
| train/             |          |
2025-12-01 01:06:34
|    actor_loss      | 4.76     |
2025-12-01 01:06:34
|    critic_loss     | 0.254    |
2025-12-01 01:06:34
|    ent_coef        | 0.00294  |
2025-12-01 01:06:34
|    ent_coef_loss   | 0.762    |
2025-12-01 01:06:34
|    learning_rate   | 0.0003   |
2025-12-01 01:06:34
|    n_updates       | 352150   |
2025-12-01 01:06:34
---------------------------------
2025-12-01 01:07:35
Training complete.
```
