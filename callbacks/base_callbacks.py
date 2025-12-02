import os
import numpy as np
import wandb # Import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
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

        # Log global_step
        metrics["global_step"] = self.num_timesteps

        # EXPLICITLY LOG TO WANDB (Fixes missing charts)
        if wandb.run is not None and metrics:
            wandb.log(metrics)

        return True


class CustomEvalCallback(EvalCallback):
    """
    Custom EvalCallback that logs evaluation metrics including portfolio (networth) to WandB.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_on_new_best = None  # Avoid init_callback error by not setting to function
        self.best_mean_portfolio = -np.inf
        self.best_std_portfolio = 0

    def _evaluate_with_portfolio(self):
        portfolio_values = []
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                action = action[0]
                obs, reward, done, info = self.eval_env.step([action])
                reward = reward[0]
                done = done[0]
                info = info[0]
                episode_reward += reward
            episode_rewards.append(episode_reward)
            portfolio_values.append(info[0].get('portfolio_value', episode_reward))  # fallback to reward if no portfolio_value
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_portfolio = np.mean(portfolio_values)
        std_portfolio = np.std(portfolio_values)
        return mean_reward, std_reward, mean_portfolio, std_portfolio

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward, mean_portfolio, std_portfolio = self._evaluate_with_portfolio()
            if self.log_path is not None:
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                self.logger.record("eval/mean_portfolio", mean_portfolio)
                self.logger.record("eval/std_portfolio", std_portfolio)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_std_reward = std_reward
                self.best_mean_portfolio = mean_portfolio
                self.best_std_portfolio = std_portfolio
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self._log_best_to_wandb(mean_reward, std_reward)
        return True

    def _log_best_to_wandb(self, mean_reward, std_reward):
        if wandb.run is not None:
            wandb.log({
                "best_eval/mean_reward": mean_reward,
                "best_eval/std_reward": std_reward,
                "best_eval/mean_portfolio": self.best_mean_portfolio,
                "best_eval/std_portfolio": self.best_std_portfolio
            }, step=self.num_timesteps)