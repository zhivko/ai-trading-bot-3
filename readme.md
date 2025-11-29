# Crypto Trader – Unsupervised Self-Improving Bot

*"Let the market teach the machine, not the human"*



## Goal

Build a fully unsupervised reinforcement-learning agent that discovers profitable crypto trading strategies by itself, with heavy emphasis on **volume profile** as the core market-structure signal.



Zero manual labeling. Zero "expert trades". The only teacher is realized PnL + risk-adjusted metrics over rolling windows.



## Core Philosophy

\- Volume Profile = institutional memory of price levels

\- High-volume nodes (HVN) = attraction zones

\- Low-volume nodes (LVN) = acceleration zones

\- The agent must rediscover these truths from raw data → true unsupervised learning



## Data Pipeline (Binance → Volume Profile Features)



```python

Timeframe:          1h candles (UTC sessions)

Pairs:              BTC-USDT, ETH-USDT, SOL-USDT (start with these)

Lookback for VP:    Rolling 7-day & 30-day profiles

Price bins:         Dynamic 0.5% bins (or fixed $25/$50 for BTC, 1% for alts)

Features per step:

   - Current VP heatmap (7d & 30d) → flattened or 2D CNN input

   - Distance of price to nearest HVN / LVN

   - Value Area High/Low (VAH/VAL) & Point of Control (POC)

   - Order-book imbalance (last 5 min)

   - Recent realized volatility

   - Time-of-day / session encoding (one-hot Asian/EU/US overlap)

Unsupervised Training Loop (Pure RL – no imitation phase)

Environment



Gym-style trading env (ccxt + custom)

Actions: Long / Short / Flat (position sizing 0–100%)


Episode length: 30–90 days rolling



Reward Function (multi-objective, evolves over time)

Start simple → get sophisticated:

textPhase 1 (weeks 1-4): Raw PnL only

Phase 2: Sortino Ratio (penalize downside vol)

Phase 3: Sortino + Calmar (max drawdown killer)

Phase 4: Add "Volume Profile Alignment Bonus"

       +λ₁ × (entry near HVN or LVN fade)

       +λ₂ × (exit near next HVN)

       -λ₃ × (fighting strong POC with size)

Phase 5 (final): Curriculum – λ weights self-tuned by meta-RL

## Current Implementation

The bot implements a Soft Actor-Critic (SAC) reinforcement learning agent with an MLP policy trained on a custom TradingEnv environment. The optimization objective is to maximize cumulative reward through profitable trading decisions.

The TradingEnv is a Gym-style environment simulating crypto trading with continuous action space. Actions are represented as a single float in [-1, 1], where positive values indicate buying (long positions) and negative values indicate selling (short positions), with magnitude controlling position size.

State observations include:
- Market features over a 30-bar lookback window: close price percentage change, normalized volume, normalized RSI, normalized Stochastic RSI, normalized MACD, and normalized MACD signal.
- Account features: normalized balance and holdings.
- Volume profile features for 7-day and 30-day rolling periods: normalized distances to POC, VAH, and VAL, plus 100-bin volume heatmaps.

The reward function incentivizes portfolio value growth while penalizing trading costs:
- Primary reward: Percentage change in net worth multiplied by 100.
- Trading penalties: 0.15% fee on executed trades, or 0.01 penalty for insignificant actions.
- Termination conditions: Episode ends if net worth drops below 50% of initial balance (bankruptcy) or reaches the end of available data.

Training occurs in an unsupervised manner, with the agent learning directly from market data and volume profile signals without any labeled examples or imitation learning. Episodes are sampled randomly from the dataset after sufficient warmup for volume profile calculation. The SAC algorithm uses entropy regularization to encourage exploration in the continuous action space, with an MLP policy network (256x256 hidden layers) and automatic entropy coefficient tuning.

## Usage

### Training
```bash
python main.py
```
Trains the SAC agent on BTC/USDT data with default settings.

### Backtesting
```bash
python backtest.py
```
Runs a backtest using the trained model and plots results.

### Visualization
```bash
python visualize_predictions.py
```
Visualizes the model's predictions and performance.

Algorithm Choice



SAC (Soft Actor-Critic) → optimized for continuous action spaces with entropy regularization

MLP policy network → processes state features without recurrence

Optional: PPO as alternative algorithm

## Reinforcement Learning Algorithms

### Soft Actor-Critic (SAC)
SAC is an off-policy actor-critic algorithm that maximizes both expected return and entropy (exploration). It uses two Q-networks for stable learning, automatic entropy coefficient tuning, and replay buffer for sample efficiency. Particularly effective for continuous action spaces like position sizing in trading.

### Proximal Policy Optimization (PPO)
PPO is an on-policy policy gradient method that uses clipped surrogate objectives to ensure stable updates. It balances exploration and exploitation through advantage estimation and is widely used for its robustness and ease of tuning across various environments.

Self-Improvement Mechanisms (no human in the loop)



Population-Based Training (PBT)

8–16 agents running in parallel, periodically copy weights + hyperparameters from best Sortino performer.

Automatic Reward Shaping

Every 7 days evaluate which VP features correlate with top-decile trades → increase their λ bonus automatically.

Risk Regime Detection

Unsupervised clustering of market states (GMM or VAE) → separate policies per regime (trending, mean-reverting, chop).

Walk-Forward Validation Only

Never train and test on same period. Fixed 30-day train → 7-day validation → deploy → repeat.





Milestones & Logging





  







MilestoneTarget SortinoNotesRandom baseline~0.0Pure noiseBeats B&H>0.8First sign of lifeVP-aware profitable>1.5Real edgeOutperforms 2021 bull run>3.0Ready for small live capital

Log everything to Weights & Biases (wandb) – rewards, VP heatmaps, equity curve, hyperparams.
