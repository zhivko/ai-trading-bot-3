import wandb
import subprocess
import os

def generate_metrics():
    api = wandb.Api()

    # Get the latest run from the project
    runs = api.runs("zhivko/ai-trading-bot")
    if runs:
        run = runs[0]  # Latest run

        # save the metrics for the run to a csv file
        metrics_dataframe = run.history()
        metrics_dataframe.to_csv("metrics.csv")
        print("Metrics saved to metrics.csv")

        # Commit and push metrics.csv
        try:
            subprocess.run(["git", "add", "metrics.csv"], check=True)
            subprocess.run(["git", "commit", "-m", "Update metrics.csv"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("metrics.csv committed and pushed to remote")
        except subprocess.CalledProcessError as e:
            print(f"Git operation failed: {e}")
    else:
        print("No runs found in the project")