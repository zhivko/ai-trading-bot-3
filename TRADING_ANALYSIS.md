# Trading Bot Analysis - Why It's Not Trading More

## Executive Summary

Your trading bot has learned to **avoid trading** rather than actively trade. This is caused by an **overly punitive reward structure** that makes "doing nothing" safer than taking action.

Looking at your evaluation chart (Image 2), the bot holds a constant short position (red bar at -1.0) throughout the entire episode with minimal trading activity.

---

## Root Causes

### 1. **Minimum Trade Value Barrier**
**Location:** `enhanced_trading_env.py`, line 335
```python
self.min_trade_value_usd = min_trade_value_usd  # Default: $10
```

**Issue:** With a $10,000 starting balance and $10 minimum trade size, the bot needs to be 0.1% confident to trade. This blocks many small position adjustments.

**Impact:** Prevents gradual position sizing and forces "all or nothing" decisions.

---

### 2. **Excessive Trading Penalties**

The bot faces **5 different penalties** for trading:

#### A. Transaction Cost Penalty (Lines 96-97, 688)
```python
self.transaction_cost_rate = 0.0015      # 0.15% per trade
self.reward_fee_multiplier = 2.0         # Magnify fee 2x
reward_trade_cost = (cost_pct * 100.0) * 2.0  # Effective: 0.30% penalty
```
**Impact:** A $1000 trade costs $3 in reward points (0.3% × $1000 × 100 scale)

#### B. Holding Penalty (Lines 99, 740)
```python
self.holding_penalty = 0.0005  # "Rent" per step
reward -= current_holding_cost  # Applied every step while holding
```
**Impact:** Discourages holding positions, even profitable ones

#### C. Action Change Penalty (Lines 701-709)
```python
action_delta = abs(action_val - self.prev_action)
action_change_penalty = -(action_delta * 0.05)
```
**Impact:** Switching from long to short (delta=2.0) costs -0.10 reward

#### D. Trend Alignment Penalty (Lines 734-735)
```python
if self.shares_held != 0 and np.sign(self.shares_held) != trend_direction:
    reward -= 0.05  # Penalty per step for fighting the trend
```
**Impact:** Punishes counter-trend trades, even if profitable

#### E. Overtrading Penalty (Lines 764-765)
```python
if trade_occurred and self.trades_in_episode > 20:
    reward -= 0.5
```
**Impact:** Hard cap at 20 trades per episode

---

### 3. **Reward Structure Analysis**

**Base Reward:** Net worth change as percentage
```python
reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 100.0
```

**Example Scenario:**
- Price moves +1% while holding long position
- Net worth increases by ~$100 (1% of $10k)
- **Base reward:** +1.0

**But then penalties apply:**
- If you just traded to enter this position:
  - Transaction cost: -0.30
  - Action change penalty: -0.05 to -0.10
  - **Net reward:** +0.60 to +0.55

**Conclusion:** The bot learns that the safest strategy is to **pick one position and hold it**, minimizing penalties.

---

## Recommendations (Ordered by Impact)

### 🔥 **HIGH PRIORITY**

#### 1. Reduce Transaction Cost Multiplier
**File:** `enhanced_trading_env.py`, line 97
```python
# BEFORE
self.reward_fee_multiplier = 2.0

# AFTER
self.reward_fee_multiplier = 1.0  # Or even 0.5 for more aggressive trading
```
**Why:** This is the biggest penalty. Cutting it in half will immediately encourage more trading.

---

#### 2. Lower Minimum Trade Value
**File:** `enhanced_trading_env.py`, line 335
```python
# BEFORE
self.min_trade_value_usd = 10.0

# AFTER
self.min_trade_value_usd = 5.0  # Or even 1.0 for micro-adjustments
```
**Why:** Allows the bot to make smaller, more frequent position adjustments.

---

#### 3. Remove or Reduce Holding Penalty
**File:** `enhanced_trading_env.py`, line 99
```python
# BEFORE
self.holding_penalty = 0.0005

# AFTER
self.holding_penalty = 0.0  # Remove entirely
# OR
self.holding_penalty = 0.0001  # Reduce by 80%
```
**Why:** This penalty contradicts the goal of profitable trading. If a position is profitable, the bot should hold it.

---

### ⚠️ **MEDIUM PRIORITY**

#### 4. Reduce Action Change Penalty
**File:** `enhanced_trading_env.py`, line 709
```python
# BEFORE
action_change_penalty = -(action_delta * 0.05)

# AFTER
action_change_penalty = -(action_delta * 0.02)  # 60% reduction
```
**Why:** The bot needs flexibility to react to market changes.

---

#### 5. Make Trend Alignment Penalty Conditional
**File:** `enhanced_trading_env.py`, lines 734-735
```python
# BEFORE
if self.shares_held != 0 and np.sign(self.shares_held) != trend_direction:
    reward -= 0.05

# AFTER
# Only penalize if BOTH fighting trend AND losing money
if self.shares_held != 0 and np.sign(self.shares_held) != trend_direction:
    if unrealized_pnl < 0:  # Only penalize if losing
        reward -= 0.05
```
**Why:** Counter-trend trades can be profitable (e.g., mean reversion). Only penalize if they're actually losing.

---

#### 6. Increase Overtrading Threshold
**File:** `enhanced_trading_env.py`, line 764
```python
# BEFORE
if trade_occurred and self.trades_in_episode > 20:
    reward -= 0.5

# AFTER
if trade_occurred and self.trades_in_episode > 50:  # 2.5x increase
    reward -= 0.5
```
**Why:** 20 trades per episode is too restrictive for an active trading strategy.

---

### 💡 **OPTIONAL ENHANCEMENTS**

#### 7. Add Reward for Profitable Trades
**File:** `enhanced_trading_env.py`, after line 760
```python
# Add this AFTER the "Closer's Bonus" section
# Reward for being in a profitable position
if self.shares_held != 0:
    unrealized_pnl_pct = ((current_price - self.entry_price) * self.shares_held) / (abs(self.entry_price * self.shares_held) + 1e-8)
    if unrealized_pnl_pct > 0:
        reward += 0.01  # Small bonus for holding a winning position
```
**Why:** Encourages the bot to enter and hold profitable positions.

---

#### 8. Reduce Inertia Penalty
**File:** `enhanced_trading_env.py`, line 694
```python
# BEFORE
if self.shares_held == self.prev_shares_held and abs(action_val) > 0.5:
    reward_inertia = -0.05

# AFTER
# Remove this penalty entirely - it's redundant with action_change_penalty
reward_inertia = 0.0
```
**Why:** This penalty is redundant and overly restrictive.

---

## Quick Fix Configuration

If you want to **quickly test** a more aggressive trading strategy, apply these changes:

```python
# enhanced_trading_env.py

# Line 97: Cut transaction penalty in half
self.reward_fee_multiplier = 1.0

# Line 99: Remove holding penalty
self.holding_penalty = 0.0

# Line 335: Lower minimum trade size
self.min_trade_value_usd = 5.0

# Line 709: Reduce action change penalty
action_change_penalty = -(action_delta * 0.02)

# Line 764: Increase overtrading threshold
if trade_occurred and self.trades_in_episode > 50:
```

---

## Expected Results

After implementing these changes, you should see:

1. ✅ **More frequent position adjustments** (10-30 trades per episode instead of 2-5)
2. ✅ **Dynamic position sizing** (gradual entries/exits instead of all-in/all-out)
3. ✅ **Better adaptation to market changes** (faster reaction to trend reversals)
4. ✅ **Higher variance in rewards** (more risk-taking behavior)

---

## Monitoring

After making changes, monitor these metrics in WandB:

- `eval/trades_per_episode` - Should increase from ~2-5 to 15-30
- `financial/sharpe_ratio` - May initially decrease (more variance) but should improve over time
- `eval/mean_portfolio` - Watch for improvement in final portfolio value
- Action distribution in logs - Should see more variety instead of constant -1.0 or +1.0

---

## Chart Fixes Applied ✅

I've also fixed the reward charting issues:

1. **TensorboardCallback** - Removed duplicate reward subplots (was plotting 3 times)
2. **CustomEvalCallback** - Added missing rewards subplot to evaluation charts

Both callbacks now properly display:
- **Subplot 1:** Price + EMA + Buy/Sell markers
- **Subplot 2:** Actions (position exposure)
- **Subplot 3:** Net Worth
- **Subplot 4:** Rewards ← NOW VISIBLE

---

## Next Steps

1. **Backup your current model** before making changes
2. **Start with the Quick Fix Configuration** above
3. **Run a short training session** (50k-100k steps) to test
4. **Monitor the evaluation charts** - you should see more trading activity
5. **Iterate:** If still too conservative, reduce penalties further
6. **If too aggressive:** Slightly increase `min_trade_value_usd` or `reward_fee_multiplier`

---

**Remember:** The goal is to find a balance where the bot trades actively enough to capitalize on opportunities, but not so much that it churns and loses money to fees.
