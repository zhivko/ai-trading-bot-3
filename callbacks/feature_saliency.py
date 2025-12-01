import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class FeatureSaliencyCallback(BaseCallback):
    """
    Analyzes which inputs drive the bot's decisions.
    Adapted for High-Dimensional Observation Spaces (Time Series + Heatmaps).
    """

    def __init__(self, vp_days, verbose=0):
        super().__init__(verbose)
        self.vp_days = vp_days
        self.log_freq = 20000 # Check every 20k steps

    def _on_step(self) -> bool:
        # 1. Skip early training or non-logging steps
        if self.num_timesteps < 50000 or self.num_timesteps % self.log_freq != 0:
            return True
        
        if wandb.run is None:
            return True

        try:
            # 2. Get the current observation (Batch size 1)
            # We grab from locals because SubprocVecEnv doesn't expose .last_obs easily
            obs = self.locals['new_obs'][0] # Take first env
            
            # Convert to Tensor
            device = self.model.device
            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0).float()
            obs_tensor.requires_grad_(True)

            # 3. Forward Pass (Get Action Distribution)
            # We want to know: "Which input changed the Action Probability the most?"
            actor = self.model.policy.actor
            dist = actor.get_action_dist_params(obs_tensor)
            
            # For SAC, we usually look at the Mean (mu) of the action distribution
            # Summing output allows backward pass
            action_mean = dist[0] # [mean, log_std]
            target = action_mean.sum() 

            # 4. Backward Pass (Calculate Gradients)
            actor.zero_grad()
            target.backward()
            
            # Get Saliency (Absolute value of gradients)
            # Shape: [Total_Features]
            grads = obs_tensor.grad.data.abs().cpu().numpy().flatten()
            
            # 5. AGGREGATE FEATURES (The Logic Fix)
            # Your env has: [Window(6*30)] + [Account(2)] + [VP(3+100)*N]
            # We need to map these thousands of numbers into readable groups.
            
            # A. Define Group Names
            std_features = ['Close %', 'Vol Norm', 'RSI', 'Stoch', 'MACD', 'Signal']
            feature_scores = {name: [] for name in std_features}
            feature_scores['Balance'] = []
            feature_scores['Shares'] = []
            
            # B. Parse the Gradient Array
            cursor = 0
            
            # -- Parse Time Series Window --
            # Assuming Lookback=30, Feats=6
            # Sequence in env: [Step1_Feat1, Step1_Feat2... Step2_Feat1...]
            # We just iterate and assign to buckets
            # Note: We need to know lookback_window size. 
            # We infer it based on standard feature count (6)
            
            # Simple heuristic: The first chunk is the window.
            # Account is 2. VP is rest.
            
            # Identify indices for Account (2 features)
            # They are usually after the window. 
            # Let's assume standard layout from trading_env.py
            
            num_std_features = 6
            # Calculate dynamic lookback based on array size
            # Formula: Total = (Lookback*6) + 2 + (VP_Days * 103)
            # This is hard to reverse, so we approximate or use hardcoded if standard.
            
            # LET'S USE A ROBUST GROUPING LOGIC
            grouped_saliency = {}
            
            # 1. Market Window (Approximate first 80% of array usually)
            # We will just group standard features by name
            # Since we flattened the window, every 6th element is the same feature type
            limit_window = len(grads) - 2 - (len(self.vp_days) * 103)
            window_grads = grads[:limit_window]
            
            for i, name in enumerate(std_features):
                # Slice every 6th element starting at i
                # e.g. indices 0, 6, 12... correspond to 'Close %'
                grouped_saliency[name] = np.mean(window_grads[i::num_std_features])

            cursor = limit_window
            
            # 2. Account
            grouped_saliency['Balance'] = grads[cursor]
            grouped_saliency['Shares'] = grads[cursor+1]
            cursor += 2
            
            # 3. Volume Profiles
            for days in self.vp_days:
                prefix = f"VP {days}d"
                # POC, VAH, VAL (3 scalars)
                grouped_saliency[f"{prefix} Levels"] = np.mean(grads[cursor:cursor+3])
                cursor += 3
                # Heatmap (100 scalars)
                grouped_saliency[f"{prefix} Heatmap"] = np.mean(grads[cursor:cursor+100])
                cursor += 100

            # 6. Plotting
            names = list(grouped_saliency.keys())
            values = list(grouped_saliency.values())
            
            # Normalize for chart
            total_importance = sum(values) + 1e-12
            values = [v / total_importance for v in values]
            
            # Sort by importance
            sorted_indices = np.argsort(values)[::-1]
            sorted_names = [names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]

            plt.figure(figsize=(12, 6))
            colors = ['orange' if x > 0.1 else '#1f77b4' for x in sorted_values]
            plt.bar(sorted_names, sorted_values, color=colors, edgecolor='black', alpha=0.8)
            plt.title(f"Brain Scan: Feature Importance (Step {self.num_timesteps})")
            plt.ylabel("Influence on Action")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            wandb.log({"analysis/feature_saliency": wandb.Image(plt)})
            plt.close()

        except Exception as e:
            # Don't crash training if visualization fails
            if self.verbose > 0:
                print(f"[Saliency Error] {e}")

        return True