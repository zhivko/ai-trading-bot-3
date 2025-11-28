\# Grok Crypto Trader – Unsupervised Self-Improving Bot  

\*“Let the market teach the machine, not the human”\*



\## Goal

Build a fully unsupervised reinforcement-learning agent that discovers profitable crypto trading strategies by itself, with heavy emphasis on \*\*volume profile\*\* as the core market-structure signal.



Zero manual labeling. Zero “expert trades”. The only teacher is realized PnL + risk-adjusted metrics over rolling windows.



\## Core Philosophy

\- Volume Profile = institutional memory of price levels  

\- High-volume nodes (HVN) = attraction zones  

\- Low-volume nodes (LVN) = acceleration zones  

\- The agent must rediscover these truths from raw data → true unsupervised learning



\## Data Pipeline (Binance → Volume Profile Features)



```python

Timeframe:          1h candles (UTC sessions)

Pairs:              BTC-USDT, ETH-USDT, SOL-USDT (start with these)

Lookback for VP:    Rolling 7-day \& 30-day profiles

Price bins:         Dynamic 0.5% bins (or fixed $25/$50 for BTC, 1% for alts)

Features per step:

&nbsp;   - Current VP heatmap (7d \& 30d) → flattened or 2D CNN input

&nbsp;   - Distance of price to nearest HVN / LVN

&nbsp;   - Value Area High/Low (VAH/VAL) \& Point of Control (POC)

&nbsp;   - Order-book imbalance (last 5 min)

&nbsp;   - Recent realized volatility

&nbsp;   - Time-of-day / session encoding (one-hot Asian/EU/US overlap)

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

Phase 4: Add “Volume Profile Alignment Bonus”

&nbsp;     +λ₁ × (entry near HVN or LVN fade) 

&nbsp;     +λ₂ × (exit near next HVN)

&nbsp;     -λ₃ × (fighting strong POC with size)

Phase 5 (final): Curriculum – λ weights self-tuned by meta-RL

## Current Implementation

In main.py, the script trains a SAC (Soft Actor-Critic) agent on the custom TradingEnv environment. The optimization objective is to maximize the expected cumulative reward over training episodes.

The reward function in TradingEnv.step() is designed to encourage profitable trading behavior with volume profile (VP) considerations:

- Primary reward: Change in portfolio value (profit/loss from position changes)
- Costs deducted: Trading fees (0.1%), slippage (0.05%), and additional fees for action changes
- Penalties:
  - Direction change penalty (-15) when switching from long to short or vice versa
  - Minimum holding penalty (-20) if exiting a position too quickly (<6 bars)
  - Penalty for large positions fighting the Point of Control (POC) in VP
- Bonuses:
  - +12 reward for long positions near VP Value Area Low (VAL)
  - +12 reward for short positions near VP Value Area High (VAH)

The agent learns to balance profitability with risk management, respecting VP signals while minimizing unnecessary trading costs and position churn.

## Usage

### Training
```bash
python main.py
```
Trains the RecurrentPPO agent on BTC/USDT data with default settings.

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

Custom LSTM features extractor → remembers multi-day profile evolution

Optional: World Models / DreamerV3 later when it starts dreaming profitable regimes



Self-Improvement Mechanisms (no human in the loop)



Population-Based Training (PBT)

8–16 agents running in parallel, periodically copy weights + hyperparameters from best Sortino performer.

Automatic Reward Shaping

Every 7 days evaluate which VP features correlate with top-decile trades → increase their λ bonus automatically.

Risk Regime Detection

Unsupervised clustering of market states (GMM or VAE) → separate policies per regime (trending, mean-reverting, chop).

Walk-Forward Validation Only

Never train and test on same period. Fixed 30-day train → 7-day validation → deploy → repeat.



Milestones \& Logging





























































MilestoneTarget SortinoNotesRandom baseline~0.0Pure noiseBeats B\&H>0.8First sign of lifeVP-aware profitable>1.5Real edgeOutperforms 2021 bull run>3.0Ready for small live capital

Log everything to Weights \& Biases (wandb) – rewards, VP heatmaps, equity curve, hyperparams.

