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
        self.dummy_env = dummy_env 

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
            
            # Handle VecEnv batching
            if len(obs.shape) > 1 and obs.shape[0] > 1:
                obs_single = obs[0:1] 
            else:
                obs_single = obs
                
            obs_tensor = torch.tensor(obs_single, dtype=torch.float32).to(self.model.device)
            obs_tensor.requires_grad_()

            # 2. Define Forward Function (Robust Architecture Check)
            def forward_func(x):
                # --- SAC LOGIC ---
                if hasattr(self.model.policy, 'actor'):
                    # SAC Policy does not have a central features_extractor, it's inside the actor
                    features = self.model.policy.actor.features_extractor(x)
                    # Get mean action
                    return self.model.policy.actor.get_action_dist_params(features)[0]
                
                # --- PPO LOGIC ---
                elif hasattr(self.model.policy, 'action_net'):
                    # PPO Policy usually has a central features_extractor
                    if self.model.policy.features_extractor is not None:
                        features = self.model.policy.features_extractor(x)
                    else:
                        # Fallback if PPO structure varies (rare)
                        features = self.model.policy.pi_features_extractor(x)
                        
                    latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                    return self.model.policy.action_net(latent_pi)
                
                else:
                    return None

            # Test it
            if forward_func(obs_tensor) is None:
                print("⚠️ Saliency: Could not determine model architecture.")
                return

            # 3. Calculate Integrated Gradients
            ig = IntegratedGradients(forward_func)
            attributions = ig.attribute(obs_tensor, baselines=torch.zeros_like(obs_tensor))
            mean_attr = attributions.mean(dim=0).cpu().detach().numpy()

            # 4. Get Names
            try:
                full_names = self.dummy_env.get_feature_names()
            except AttributeError:
                full_names = [f"Feat_{i}" for i in range(len(mean_attr))]
            
            # Match lengths
            min_len = min(len(full_names), len(mean_attr))
            full_names = full_names[:min_len]
            mean_attr = mean_attr[:min_len]

            # 5. Aggregation
            agg_map = {}
            for name, val in zip(full_names, mean_attr):
                imp = abs(val)
                # Clean up names for grouping
                if "VP_" in name and "Bin" in name: 
                    group = "Volume Profile (Heatmap)"
                elif "VP_" in name and "Dist" in name: 
                    group = "VP Levels"
                elif "_t-" in name: 
                    group = name.split('_t-')[0].replace('_norm', '') + " (Window)"
                else: 
                    group = name.replace('_norm', '')
                
                agg_map[group] = agg_map.get(group, 0.0) + imp

            self.plot_importance(list(agg_map.keys()), list(agg_map.values()), self.num_timesteps)
            
        except Exception as e:
            print(f"⚠️ Saliency Error during calculation: {e}")
            # Optional: Print traceback to debug further if needed
            # import traceback; traceback.print_exc()

    def plot_importance(self, names, values, step):
        # Sort desc
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