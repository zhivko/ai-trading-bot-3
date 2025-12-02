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
            if isinstance(obs, dict): obs = obs['default']
            
            # Take just the first observation if using multiple envs
            if len(obs.shape) > 1 and obs.shape[0] > 1:
                obs_single = obs[0:1] # Keep batch dim [1, features]
            else:
                obs_single = obs
                
            obs_tensor = torch.tensor(obs_single, dtype=torch.float32).to(self.model.device)
            obs_tensor.requires_grad_()

            # 2. Define Forward Function
            def forward_func(x):
                features = self.model.policy.extract_features(x)
                if hasattr(self.model.policy, 'actor'): # SAC
                    return self.model.policy.actor.get_action_dist_params(features)[0]
                elif hasattr(self.model.policy, 'action_net'): # PPO
                    latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                    return self.model.policy.action_net(latent_pi)
                else:
                    return None

            if forward_func(obs_tensor) is None:
                return

            # 3. Calculate IG
            ig = IntegratedGradients(forward_func)
            attributions = ig.attribute(obs_tensor, baselines=torch.zeros_like(obs_tensor))
            mean_attr = attributions.mean(dim=0).cpu().detach().numpy()

            # 4. Get Names & Aggregate
            full_names = self.dummy_env.get_feature_names()
            
            # Ensure lengths match
            min_len = min(len(full_names), len(mean_attr))
            full_names = full_names[:min_len]
            mean_attr = mean_attr[:min_len]

            agg_map = {}
            for name, val in zip(full_names, mean_attr):
                imp = abs(val)
                if "VP_" in name and "Bin" in name: group = "Volume Profile (Heatmap)"
                elif "VP_" in name and "Dist" in name: group = "VP Levels"
                elif "_t-" in name: group = name.split('_t-')[0] # Collapse windowed features
                else: group = name
                
                agg_map[group] = agg_map.get(group, 0.0) + imp

            # Plot
            self.plot_importance(list(agg_map.keys()), list(agg_map.values()), self.num_timesteps)
            
        except Exception as e:
            print(f"⚠️ Saliency Error: {e}")

    def plot_importance(self, names, values, step):
        zipped = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
        names, values = zip(*zipped)
        
        plt.figure(figsize=(12, 6))
        plt.bar(names, values, color='#4caf50', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Brain Scan: Feature Importance (Step {step})")
        plt.ylabel("Influence")
        plt.tight_layout()
        
        if wandb.run is not None:
            wandb.log({"feature_importance": wandb.Image(plt)}, step=step)
        plt.close()