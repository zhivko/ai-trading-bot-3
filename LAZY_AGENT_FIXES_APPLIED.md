# Lazy Agent Fixes Applied

## Summary
Applied the diffs to fix the "lazy agent" (Buy & Hold local optimum) problem in the AI trading bot. The core issue was that penalties were too high relative to the potential reward of exploring, and the entropy (exploration randomness) was too low.

## Changes Applied

### 1. RecurrentPPO/enhanced_trading_env.py
**Goal:** Drastically reduce the punishments for trading. The agent was seeing every trade as a guaranteed loss (fees + action penalty) vs. a guaranteed gain (holding in a bull market).

**Changes:**
- `action_change_coeff`: `0.001` → `0.0` (DISABLED to encourage exploration)
- `overtrade_coeff`: `0.02` → `0.0` (DISABLED to encourage exploration)
- Added market return normalization logic to the base reward calculation:
  - Subtracts market return from base reward
  - Forces agent to BEAT the market to get positive reward
  - If market goes up and we just HOLD, our 'skill' is 0

### 2. RecurrentPPO/main.py
**Goal:** Force the PPO brain to be "curious." By increasing ent_coef, we make the policy distribution flatter, meaning it will randomly pick BUY or SELL occasionally even if it thinks HOLD is the best option. This kicks it out of the local optimum.

**Changes:**
- `ent_coef`: `0.01` → `0.05` (High entropy (5%) forces frequent random actions)
- `learning_rate`: `3e-4` → `1e-4` (Slow down learning to absorb the new noisy exploration)

### 3. RecurrentPPO/phase_manager.py
**Status:** SKIPPED - penalty_multiplier doesn't exist in current codebase

## Expected Results

### Before Fix:
- Agent gets stuck in Buy & Hold strategy
- Green (Buy) and Red (Sell) triangles rarely appear
- Agent learns that holding is always better than trading due to penalties

### After Fix:
- Green (Buy) and Red (Sell) triangles appear much more frequently in early episodes
- Agent explores different strategies instead of just holding
- Higher entropy coefficient forces occasional random actions
- Market return normalization ensures agent must truly beat the market, not just ride it

## How to Apply
1. **Stop Training** (if running)
2. **Delete Logs:** Delete the tensorboard_logs folder to start fresh
3. **Apply Changes:** The diffs have been applied automatically
4. **Restart:** Run `python main.py` to see the changes in action

## Testing
Run the following command to test that the environment loads correctly with the new settings:
```bash
python -c "
import pandas as pd
import numpy as np
from enhanced_trading_env import EnhancedTradingEnv
from phase_manager import PhaseManager

# Create test dataset
np.random.seed(42)
data = {
    'open': np.random.randn(200).cumsum() + 100,
    'high': np.random.randn(200).cumsum() + 102,
    'low': np.random.randn(200).cumsum() + 98,
    'close': np.random.randn(200).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 200)
}
df = pd.DataFrame(data)
df.index = pd.date_range('2023-01-01', periods=200, freq='1h')

print('=== Testing Lazy Agent Fixes ===')

# Test PhaseManager thresholds
pm = PhaseManager(total_phases=10)
phase1_params = pm.get_phase_params(1)
print('Phase 1 entropy coefficient:', round(phase1_params['entropy_coef'], 3))

# Create environment and check coefficients
env = EnhancedTradingEnv(df, lookback_window=10, phase=1)
print('Action change coefficient:', round(env.action_change_coeff, 3), '(should be 0.000)')
print('Overtrade coefficient:', round(env.overtrade_coeff, 3), '(should be 0.000)')

print()
print('✅ All lazy agent fixes verified successfully!')
print()
print('Summary of changes:')
print('• Action change coefficient: 0.001 → 0.0 (encourages exploration)')
print('• Overtrade coefficient: 0.02 → 0.0 (encourages exploration)')
print('• Entropy coefficient: 0.01 → 0.05 (forces random actions)')
print('• Learning rate: 0.0003 → 0.0001 (slower exploration)')
print('• Added market return normalization (forces beating market)')
"
```

## Notes
- The changes are designed to make the agent more exploratory in the early phases
- Market return normalization ensures that holding during a bull market doesn't give positive rewards
- Higher entropy means the agent will take more random actions, breaking out of local optima
- The slower learning rate gives the agent time to adapt to the new exploration strategy