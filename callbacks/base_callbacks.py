import os
import numpy as np
import wandb # Import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in Tensorboard AND WandB.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if multiple envs (SubprocVecEnv) or single (DummyVecEnv)
        infos = self.locals.get("infos", [{}])
        if isinstance(infos, list) and len(infos) > 0:
            info = infos[0] # Log the first env's info
        else:
            info = {}

        # Prepare metrics dictionary
        metrics = {}
        
        if "portfolio_value" in info:
            val = info["portfolio_value"]
            self.logger.record("market_context/portfolio_value", val)
            metrics["market_context/portfolio_value"] = val
            
        if "balance" in info:
            val = info["balance"]
            self.logger.record("market_context/balance", val)
            metrics["market_context/balance"] = val
            
        if "price" in info:
            val = info["price"]
            self.logger.record("market_context/price_main", val)
            metrics["market_context/price_main"] = val
        
        # Log Action Distribution (Mean)
        actions = self.locals.get("actions", None)
        if actions is not None:
            val = np.mean(actions)
            self.logger.record("action/action_mean", val)
            metrics["action/action_mean"] = val

        # EXPLICITLY LOG TO WANDB (Fixes missing charts)
        if wandb.run is not None and metrics:
            wandb.log(metrics, step=self.num_timesteps)

        return True