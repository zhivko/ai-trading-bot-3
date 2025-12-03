import wandb

api = wandb.Api()

# Get the latest run from the project
runs = api.runs("zhivko/ai-trading-bot")
if runs:
    run = runs[0]  # Latest run

    # save the metrics for the run to a csv file
    metrics_dataframe = run.history()
    metrics_dataframe.to_csv("metrics.csv")
    print("Metrics saved to metrics.csv")
else:
    print("No runs found in the project")