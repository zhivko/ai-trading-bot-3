# Quick Fix - Increase Trading Activity

## Apply These Changes to `enhanced_trading_env.py`

### 1. Line 97 - Reduce Transaction Cost Multiplier
```python
# CHANGE FROM:
self.reward_fee_multiplier = 2.0

# CHANGE TO:
self.reward_fee_multiplier = 1.0  # 50% reduction
```

### 2. Line 99 - Remove Holding Penalty
```python
# CHANGE FROM:
self.holding_penalty = 0.0005

# CHANGE TO:
self.holding_penalty = 0.0  # Removed
```

### 3. Line 335 - Lower Minimum Trade Value
```python
# CHANGE FROM:
self.min_trade_value_usd = min_trade_value_usd  # Default: 10.0

# CHANGE TO:
self.min_trade_value_usd = 5.0  # Allow smaller trades
```

### 4. Line 709 - Reduce Action Change Penalty
```python
# CHANGE FROM:
action_change_penalty = -(action_delta * 0.05)

# CHANGE TO:
action_change_penalty = -(action_delta * 0.02)  # 60% reduction
```

### 5. Line 764 - Increase Overtrading Threshold
```python
# CHANGE FROM:
if trade_occurred and self.trades_in_episode > 20:

# CHANGE TO:
if trade_occurred and self.trades_in_episode > 50:  # 2.5x increase
```

### 6. Line 694 - Remove Inertia Penalty (Optional)
```python
# CHANGE FROM:
if self.shares_held == self.prev_shares_held and abs(action_val) > 0.5:
     reward_inertia = -0.05

# CHANGE TO:
reward_inertia = 0.0  # Remove this penalty entirely
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Trades per Episode | 2-5 | 15-30 |
| Transaction Penalty | -0.30% | -0.15% |
| Position Flexibility | Low | High |
| Holding Cost | -0.05/step | 0.0 |
| Max Trades | 20 | 50 |

---

## Testing

1. **Backup current model:** `cp -r models/ models_backup/`
2. **Apply changes above**
3. **Run short training:** 50k-100k steps
4. **Check WandB metrics:**
   - `eval/trades_per_episode` should increase
   - Action distribution should show more variety
   - Net worth should remain stable or improve

---

## Rollback if Needed

If the bot becomes too aggressive (overtrading):
- Increase `min_trade_value_usd` to 7.5
- Increase `reward_fee_multiplier` to 1.5
- Reduce `trades_in_episode` threshold to 35

---

## Chart Fixes (Already Applied ✅)

The reward charts in both callbacks now display correctly:
- **TensorboardCallback:** Fixed duplicate reward subplots
- **CustomEvalCallback:** Added missing rewards subplot

Both now show 4 subplots: Price/EMA, Actions, Net Worth, **Rewards**
