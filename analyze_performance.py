import json
import sys

def analyze_performance(data):
    """
    Analyze trading bot performance from WandB metrics JSON.
    """
    print("=== TRADING BOT PERFORMANCE ANALYSIS ===\n")

    # Evaluation Metrics
    print("EVALUATION METRICS (Test Performance):")
    eval_ep_length = data.get("eval/mean_ep_length", 0)
    eval_portfolio_value = data.get("eval/mean_portfolio_value", 0)
    eval_reward = data.get("eval/mean_reward", 0)
    eval_portfolio_final = data.get("eval/portfolio_value", 0)

    print(f"  Mean Episode Length: {eval_ep_length:.0f} steps")
    print(f"  Mean Portfolio Value: ${eval_portfolio_value:,.0f}")
    print(f"  Mean Reward: {eval_reward:.2f}")
    print(f"  Final Portfolio Value: ${eval_portfolio_final:,.0f}")

    if eval_portfolio_value <= 10000:
        eval_performance = "Poor - No profit in evaluation"
    elif eval_portfolio_value < 15000:
        eval_performance = "Moderate - Some profit but limited"
    else:
        eval_performance = "Good - Significant profit in evaluation"
    print(f"  Assessment: {eval_performance}\n")

    # Real-time Metrics
    print("REAL-TIME METRICS (Training Performance):")
    rt_portfolio = data.get("realtime/portfolio_value", 0)
    rt_balance = data.get("realtime/balance", 0)
    rt_reward = data.get("realtime/step_reward", 0)
    rt_action = data.get("realtime/action", 0)
    rt_alpha = data.get("realtime/alpha_entropy", 0)
    rt_lr = data.get("realtime/learning_rate", 0)

    print(f"  Portfolio Value: ${rt_portfolio:,.2f}")
    print(f"  Balance: ${rt_balance:.2f}")
    print(f"  Step Reward: {rt_reward:.2f}")
    print(f"  Current Action: {rt_action:.2f} ({'Long' if rt_action > 0.2 else 'Short' if rt_action < -0.2 else 'Neutral'})")
    print(f"  Alpha Entropy: {rt_alpha:.4f}")
    print(f"  Learning Rate: {rt_lr}")

    if rt_portfolio > 100000:
        rt_performance = "Excellent - Strong profits in training"
    elif rt_portfolio > 20000:
        rt_performance = "Good - Decent profits"
    else:
        rt_performance = "Warning: Limited - Minimal profits"
    print(f"  Assessment: {rt_performance}\n")

    # Market Context
    print("MARKET CONTEXT:")
    price_main = data.get("realtime/market_context.realtime/price_main", 0)
    price_poc = data.get("realtime/market_context.realtime/price_poc", 0)
    price_vah = data.get("realtime/market_context.realtime/price_vah", 0)
    price_val = data.get("realtime/market_context.realtime/price_val", 0)

    print(f"  Current Price: ${price_main:.0f}")
    print(f"  POC (Point of Control): ${price_poc:.0f}")
    print(f"  VAH (Value Area High): ${price_vah:.0f}")
    print(f"  VAL (Value Area Low): ${price_val:.0f}")

    if price_main > price_vah:
        position = "Above Value Area"
    elif price_main < price_val:
        position = "Below Value Area"
    else:
        position = "Within Value Area"
    print(f"  Price Position: {position}\n")

    # Training Progress
    global_step = data.get("global_step", 0)
    print("TRAINING PROGRESS:")
    print(f"  Global Step: {global_step:,}")
    print(".1f")

    # Overall Assessment
    print("OVERALL ASSESSMENT:")
    if rt_portfolio > 100000 and eval_portfolio_value > 10000:
        overall = "Strong Performance - Good generalization"
    elif rt_portfolio > 100000 and eval_portfolio_value <= 10000:
        overall = "Warning: Overfitting Suspected - Great in training, poor in eval"
    elif rt_portfolio < 20000:
        overall = "Error: Needs Improvement - Poor performance overall"
    else:
        overall = "Mixed Results - Check hyperparameters or data"
    print(f"  {overall}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if eval_portfolio_value <= 10000 and rt_portfolio > 100000:
        print("  - Investigate overfitting: Try regularization, more diverse data, or adjust eval frequency")
        print("  - Consider early stopping or better validation")
    if rt_reward < 0:
        print("  - Negative rewards: Review reward function or environment setup")
    if abs(rt_action) < 0.1:
        print("  - Low action magnitude: Model may be too conservative")
    if rt_alpha < 0.01:
        print("  - Low entropy: SAC may be exploiting too much, try higher ent_coef")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Load from file
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
    else:
        # Use the provided JSON data
        data = {
            "_runtime": 9436.811489582062,
            "_step": 5413,
            "_timestamp": 1764461863.8505037,
            "eval/mean_ep_length": 11414,
            "eval/mean_portfolio_value": 10000,
            "eval/mean_reward": -83.198512,
            "eval/portfolio_value": 10000,
            "global_step": 1063300,
            "realtime/action": 0.43618476390838623,
            "realtime/alpha_entropy": 0.0031255289062925673,
            "realtime/balance": 12.744748471912716,
            "realtime/learning_rate": 0,
            "realtime/market_context.realtime/price_main": 8848,
            "realtime/market_context.realtime/price_poc": 8851.9365234375,
            "realtime/market_context.realtime/price_vah": 9129.7138671875,
            "realtime/market_context.realtime/price_val": 7676.72509765625,
            "realtime/portfolio_value": 1212522.3310518018,
            "realtime/step_reward": 0.7633360033449851,
            "realtime/vp_snapshot_table": {
                "_latest_artifact_path": "wandb-client-artifact://9fgrmrdunylpns3ty773ru5ogq00ewn455dp7gja3udos218kvd6tbmwntksnh0dutrjrnrr7n7is4r7qbocy118l86c2ybzdpqz6wsqtm7ax5rtzyuj10qe80185yez:latest/realtime/vp_snapshot_table.table.json",
                "_type": "table-file",
                "artifact_path": "wandb-client-artifact://brvex60zt30z71abaj3i11dxfu6buvmwiupo64vwzbz9qotruw7u1uqb5xbddjsfhj45ql07px0x6lbu7cokwqm6v0zh7sth6en6kzm5avimgekli47jlxw2brgagwu3/realtime/vp_snapshot_table.table.json",
                "ncols": 3,
                "nrows": 100,
                "path": "media/table/realtime/vp_snapshot_table_5400_33311365861b8fbf7932.table.json",
                "sha256": "33311365861b8fbf7932954bc2c3af50e2b3f068d1e07261eeb495d76f55398c",
                "size": 4637
            }
        }

    analyze_performance(data)