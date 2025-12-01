# callbacks/feature_saliency.py   ← new file
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# These are the exact 23 features in the order your features.py returns them
FEATURE_NAMES = [
    "rsi_14",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "dist_to_poc_7d",
    "dist_to_vah_7d",
    "dist_to_val_7d",
    "hvn_ratio_7d",
    "lvn_ratio_7d",
    "dist_to_poc_30d",
    "dist_to_vah_30d",
    "dist_to_val_30d",
    "hvn_ratio_30d",
    "lvn_ratio_30d",
    "price_zscore_24h",
    "volume_zscore_24h",
    "session_asia",
    "session_eu",
    "session_us",
    "hour_sin",
    "hour_cos",
]

class FeatureSaliencyCallback(BaseCallback):
    """
    Logs a clean bar chart to W&B showing which of your 23 features
    actually drives the policy the most.
    Fully compatible with your current VecEnv + features.py
    """

    def __init__(self, log_freq: int = 25000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < 150_000:        # wait until policy is decent
            return True

        if self.num_timesteps % self.log_freq != 0:
            return True

        try:
            # Get the very latest observation from any environment (they're all synced)
            obs = self.training_env.envs[0].last_obs  # numpy array (23,)
            if obs is None:
                return True

            obs_tensor = torch.FloatTensor(obs).to(self.model.device).unsqueeze(0)
            obs_tensor.requires_grad_(True)

            # Forward through actor
            distribution = self.model.policy.get_distribution(obs_tensor)
            action = distribution.rsample()
            log_prob = distribution.log_prob(action).sum(-1)

            # Use episode return (or Sortino if you log it) as scalar multiplier
            info = self.locals["infos"][0]
            episode_return = info.get("episode", {}).get("r", 1.0)
            sortino = info.get("sortino", episode_return)  # fallback
            scalar = max(sortino, 0.1)

            # Backprop
            (log_prob * scalar).backward()
            saliency = obs_tensor.grad.abs().cpu().numpy().flatten()
            saliency = saliency / (saliency.sum() + 1e-12)

            # Plot
            plt.figure(figsize=(14, 5))
            colors = ['orange' if x > np.percentile(saliency, 85) else '#1f77b4' for x in saliency]
            bars = plt.bar(FEATURE_NAMES, saliency, color=colors, edgecolor='black', linewidth=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Normalized |∇_obs| (influence)")
            plt.title(f"Feature Saliency — Step {self.num_timesteps:,} | Ep.Return ≈ {episode_return:.1f}")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            wandb.log({"feature_saliency": wandb.Image(plt)})
            plt.close()

        except Exception as e:
            if self.verbose > 0:
                print(f"[Saliency] {e}")

        return True