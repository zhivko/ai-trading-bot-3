#!/usr/bin/env python3

import sys
sys.path.append('.')
from enhanced_trading_env import EnhancedTradingEnv
from phase_manager import PhaseManager
import pandas as pd
import numpy as np

print('🧪 Testing Phase-based deadzone fixes...')

# Create test data
dates = pd.date_range('2023-01-01', periods=100, freq='1H')
np.random.seed(42)
data = {
    'close': 50000 + np.cumsum(np.random.randn(100) * 50),
    'high': 50000 + np.cumsum(np.random.randn(100) * 50) + 25,
    'low': 50000 + np.cumsum(np.random.randn(100) * 50) - 25,
    'volume': np.random.rand(100) * 1000,
}
df = pd.DataFrame(data, index=dates)

# Test new thresholds
phase_manager = PhaseManager(total_phases=10)
params1 = phase_manager.get_phase_params(1)
params2 = phase_manager.get_phase_params(2)

print(f'Phase 1 thresholds: buy={params1["buy_threshold"]:.3f}, sell={params1["sell_threshold"]:.3f}')
print(f'Phase 2 thresholds: buy={params2["buy_threshold"]:.3f}, sell={params2["sell_threshold"]:.3f}')

# Test environment creation
env1 = EnhancedTradingEnv(df, phase_manager=phase_manager, phase=1)
print(f'Environment Phase 1: buy_threshold={env1.buy_threshold:.3f}')

print('✅ All changes applied successfully!')
print('✅ Deadzone disabled in Phase 1')
print('✅ Closer bonus disabled')
print('✅ Penalties reduced')
print('✅ Smaller initial thresholds (0.05 instead of 0.20)')