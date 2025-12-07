import os
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from captum.attr import IntegratedGradients
import logging

class RecurrentFeatureSaliencyCallback(BaseCallback):
    """
    Computes Feature Saliency (Integrated Gradients) for RecurrentPPO.
    Handles the LSTM hidden states injection required by the policy.
    """
    def __init__(
        self,
        check_freq: int = 5000,
        save_path: str = "./logs/saliency",
        feature_names: list = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.feature_names = feature_names
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.compute_saliency()
        return True

    def compute_saliency(self):
        """
        Computes gradients of the action output with respect to the input features.
        """
        # CRITICAL FIX: CuDNN LSTMs require training mode to compute gradients (backward pass)
        # We save the current state to restore it later.
        was_training = self.model.policy.training
        self.model.policy.train()

        try:
            # 1. Get current observation and convert to Tensor
            # SB3 normalizes envs, so we use the stored 'new_obs' from locals
            obs_array = self.locals['new_obs']  # Shape: (n_envs, n_features)

            # If using VecNormalize, this is already normalized.
            # If n_envs > 1, we just take the first environment for visualization.
            obs_tensor = th.as_tensor(obs_array[0:1]).float().to(self.model.device)
            obs_tensor.requires_grad = True

            # 2. Get Current LSTM States and Episode Starts
            # RecurrentPPO stores states in _last_lstm_states as RNNStates(pi=(h, c), vf=(h, c))
            lstm_states = self.model._last_lstm_states

            # We need the Policy (Actor) states, which are at index 0 (or .pi)
            # lstm_states[0] is the tuple (hidden_state, cell_state) for the Actor
            actor_states = lstm_states[0]

            # Now we slice the TENSORS inside that tuple
            single_env_lstm_states = (
                actor_states[0][:, 0:1, :].clone(), # Hidden State Tensor
                actor_states[1][:, 0:1, :].clone()  # Cell State Tensor
            )

            # Episode start flag (usually False in middle of episode)
            episode_starts = self.model._last_episode_starts[0:1]
            episode_starts_tensor = th.tensor(episode_starts).float().to(self.model.device)

            # 3. Define the Wrapper Function for Captum
            # Captum expects: func(inputs) -> output
            # It creates a batch of interpolated inputs (e.g., 50 steps), so we must
            # expand our LSTM states to match that batch size.
            def forward_func(inputs):
                batch_size = inputs.shape[0]

                # Repeat LSTM states to match Captum's internal batch size
                # (n_layers, 1, hidden) -> (n_layers, batch_size, hidden)
                h_expanded = single_env_lstm_states[0].repeat(1, batch_size, 1)
                c_expanded = single_env_lstm_states[1].repeat(1, batch_size, 1)
                expanded_states = (h_expanded, c_expanded)

                # Repeat episode starts
                starts_expanded = episode_starts_tensor.repeat(batch_size)

                # Get Action Distribution
                # RecurrentPPO returns a tuple: (distribution, new_states)
                # We must unpack it to get the actual distribution object.
                results = self.model.policy.get_distribution(inputs, expanded_states, starts_expanded)

                # Handle both cases (Tuple vs Object) for robustness
                if isinstance(results, tuple):
                    distribution = results[0]
                else:
                    distribution = results

                # For Continuous actions (Box), mode/mean is the action.
                # If multiple actions, we sum them (magnitude) or pick the first dimension.
                # Summing is a good proxy for "Total Action Impact".
                action_mean = distribution.mode()
                return action_mean.sum(dim=1)

            # 4. Compute Integrated Gradients
            ig = IntegratedGradients(forward_func)
        
            try:
                # Check baseline (zero vector)
                baseline = th.zeros_like(obs_tensor)

                # OPTIMIZATION: n_steps=10 (Default is 50)
                # This makes the calculation 5x FASTER.
                # 10 steps is usually enough to identify major features without freezing training.
                attributions, delta = ig.attribute(
                    obs_tensor,
                    baselines=baseline,
                    n_steps=10,
                    return_convergence_delta=True
                )

                # 5. Process and Save Results
                attrs = attributions.detach().cpu().numpy()[0]

                # Create DataFrame
                if self.feature_names and len(self.feature_names) == len(attrs):
                    df_attrs = pd.DataFrame({'Feature': self.feature_names, 'Importance': attrs})
                else:
                    df_attrs = pd.DataFrame({'Feature': [f'F_{i}' for i in range(len(attrs))], 'Importance': attrs})

                # Sort by absolute importance
                df_attrs['Abs_Importance'] = df_attrs['Importance'].abs()
                df_attrs = df_attrs.sort_values('Abs_Importance', ascending=False).head(20)

                # Plot
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=df_attrs, palette='viridis')
                plt.title(f"RecurrentPPO Feature Saliency (Step {self.n_calls})")
                plt.tight_layout()

                # --- FIX: Log to WandB ---
                try:
                    wandb.log({"Explainability/Feature_Saliency": wandb.Image(plt.gcf())}, commit=False)
                except Exception as e:
                    logging.info(f"[Saliency] WandB Log Failed: {e}")
                # -------------------------

                plt.savefig(os.path.join(self.save_path, f"saliency_step_{self.n_calls}.png"))
                plt.close()

                if self.verbose > 0:
                    logging.info(f"[Saliency] Top feature: {df_attrs.iloc[0]['Feature']} ({df_attrs.iloc[0]['Importance']:.4f})")

            except Exception as e:
                logging.info(f"[Saliency] Error computing gradients: {e}")

        finally:
            # CRITICAL: Always restore the original mode (likely Eval if inside callback)
            # to prevent messing up the actual training loop or evaluation.
            self.model.policy.train(was_training)