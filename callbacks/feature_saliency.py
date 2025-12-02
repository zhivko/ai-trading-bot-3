import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from captum.attr import IntegratedGradients

class FeatureSaliencyCallback(BaseCallback):
    def __init__(self, dummy_env, verbose=0, check_freq=50000):
        """
        :param dummy_env: A SINGLE instance of TradingEnv (not VecEnv) used to get feature names.
        """
        super(FeatureSaliencyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.dummy_env = dummy_env # Store the single env instance

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.explain_model()
        return True

    def explain_model(self):
        try:
            # 1. Prepare Observation
            obs = self.locals['new_obs']
            if isinstance(obs, dict): 
                obs = obs['default']
            
            # Handle VecEnv batching (take 1st item)
            if len(obs.shape) > 1 and obs.shape[0] > 1:
                obs_single = obs[0:1] # Keep batch dim [1, features]
            else:
                obs_single = obs
                
            obs_tensor = torch.tensor(obs_single, dtype=torch.float32).to(self.model.device)
            obs_tensor.requires_grad_()

            # 2. Define Forward Function (FIXED: Bypassing extract_features helper)
            def forward_func(x):
                # Direct call to the PyTorch module
                features = self.model.policy.features_extractor(x)
                
                # Logic for SAC vs PPO architectures
                if hasattr(self.model.policy, 'actor'): # SAC
                    # get_action_dist_params returns (mean_actions, log_std)
                    return self.model.policy.actor.get_action_dist_params(features)[0]
                
                elif hasattr(self.model.policy, 'action_net'): # PPO
                    # PPO requires an intermediate projection (mlp_extractor)
                    latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                    return self.model.policy.action_net(latent_pi)
                
                else:
                    return None

            # Test if function works before running heavy calculation
            if forward_func(obs_tensor) is None:
                print("⚠️ Saliency: Could not determine model architecture.")
                return

            # 3. Calculate Integrated Gradients
            ig = IntegratedGradients(forward_func)
            attributions = ig.attribute(obs_tensor, baselines=torch.zeros_like(obs_tensor))
            mean_attr = attributions.mean(dim=0).cpu().detach().numpy()

            # 4. Get Names & Aggregate
            try:
                full_names = self.dummy_env.get_feature_names()
            except AttributeError:
                full_names = [f"Feat_{i}" for i in range(len(mean_attr))]
            
            # Ensure lengths match
            min_len = min(len(full_names), len(mean_attr))
            full_names = full_names[:min_len]
            mean_attr = mean_attr[:min_len]

            # 5. Aggregation Logic (Clean up the chart)
            agg_map = {}
            for name, val in zip(full_names, mean_attr):
                imp = abs(val)
                if "VP_" in name and "Bin" in name: 
                    group = "Volume Profile (Heatmap)"
                elif "VP_" in name and "Dist" in name: 
                    group = "VP Levels"
                elif "_t-" in name: 
                    # Collapse windowed features (e.g. rsi_t-0, rsi_t-1 -> rsi)
                    group = name.split('_t-')[0]
                    # Also remove common suffixes like _norm if you want cleaner labels
                    group = group.replace('_norm', '')
                else: 
                    group = name.replace('_norm', '')
                
                agg_map[group] = agg_map.get(group, 0.0) + imp

            # Plot
            self.plot_importance(list(agg_map.keys()), list(agg_map.values()), self.num_timesteps)
            
        except Exception as e:
            print(f"⚠️ Saliency Error during calculation: {e}")
            import traceback
            traceback.print_exc()

    def plot_importance(self, names, values, step):
        # Sort by importance
        zipped = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
        names, values = zip(*zipped)
        
        plt.figure(figsize=(12, 6))
        plt.bar(names, values, color='#4caf50', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Brain Scan: Feature Importance (Step {step})")
        plt.ylabel("Influence on Action")
        plt.tight_layout()
        
        if wandb.run is not None:
            wandb.log({"feature_importance": wandb.Image(plt)}, step=step)
        plt.close()