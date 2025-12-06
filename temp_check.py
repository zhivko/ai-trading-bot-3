import numpy as np
import logging

# Test clipping
action = np.array([3.0])
print(f"Original action: {action}")

clipped = np.clip(action, -1.0, 1.0)
print(f"Clipped action: {clipped}")

action_val = float(clipped[0])
print(f"Action val: {action_val}")