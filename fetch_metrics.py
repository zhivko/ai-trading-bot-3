import wandb
import subprocess
import os
import time
import pandas as pd
import plotly.graph_objects as go
import threading

def calculate_financial_kpis(history_df, summary_dict):
    """
    Derive financial metrics using Summary (latest) and History (trends).
    """
    kpis = {}
    
    # --- 1. Reward (Evaluation vs Training) ---
    if 'best_eval/mean_reward' in summary_dict:
        kpis["Final Eval Reward"] = f"{summary_dict['best_eval/mean_reward']:.2f}"
    elif 'rollout/ep_rew_mean' in history_df.columns:
        val = history_df['rollout/ep_rew_mean'].dropna()
        if not val.empty:
            kpis["Final Mean Reward"] = f"{val.iloc[-1]:.2f}"
        else:
            kpis["Final Reward"] = "N/A"
    else:
        kpis["Final Reward"] = "N/A"

    # --- 2. Total Steps (Fix for incorrect step count) ---
    # Priority: time/total_timesteps (SB3) -> global_step (WandB) -> _step
    if 'time/total_timesteps' in summary_dict:
        total = summary_dict['time/total_timesteps']
    elif 'time/total_timesteps' in history_df.columns:
        total = history_df['time/total_timesteps'].max()
    elif 'global_step' in history_df.columns:
        total = history_df['global_step'].max()
    else:
        total = summary_dict.get('_step', 0)
    
    kpis["Total Env Steps"] = f"{total:,.0f}"

    # --- 3. Training Time ---
    time_keys = ['time/time_elapsed', '_runtime']
    time_val = None
    for key in time_keys:
        if key in summary_dict:
            time_val = summary_dict[key]
            break
        elif key in history_df.columns:
            time_val = history_df[key].max()
            break
    
    if time_val:
        kpis["Training Duration"] = f"{float(time_val) / 3600:.1f} Hours"

    # --- 4. Financial Metrics (From new Callback) ---
    # Sharpe Ratio
    sharpe = summary_dict.get('financial/sharpe_ratio')
    if sharpe is not None:
        kpis["Sharpe Ratio"] = f"{sharpe:.2f}"

    # Sortino Ratio
    sortino = summary_dict.get('financial/sortino_ratio')
    if sortino is not None:
        kpis["Sortino Ratio"] = f"{sortino:.2f}"

    # Calmar Ratio
    calmar = summary_dict.get('financial/calmar_ratio')
    if calmar is not None:
        kpis["Calmar Ratio"] = f"{calmar:.2f}"

    # Max Drawdown
    mdd = summary_dict.get('financial/max_drawdown')
    if mdd is not None:
        kpis["Max Drawdown"] = f"<span style='color:red'>{mdd:.2%}</span>"

    # Annualized Return
    ann_ret = summary_dict.get('financial/annualized_return')
    if ann_ret is not None:
        color = "green" if ann_ret > 0 else "red"
        kpis["Annualized Return"] = f"<span style='color:{color}'>{ann_ret:.2%}</span>"

    # --- 5. Benchmark Comparison (Alpha) ---
    strat_ret = summary_dict.get('financial/strategy_return')
    bench_ret = summary_dict.get('financial/benchmark_return')
    
    if strat_ret is not None:
        color = "green" if strat_ret > 0 else "red"
        kpis["Strategy Return"] = f"<span style='color:{color}'>{strat_ret:.2%}</span>"
        
    if bench_ret is not None:
        color = "green" if bench_ret > 0 else "red"
        kpis["Buy & Hold"] = f"<span style='color:{color}'>{bench_ret:.2%}</span>"

    if strat_ret is not None and bench_ret is not None:
        alpha = strat_ret - bench_ret
        color = "#00ff00" if alpha > 0 else "#ff4444" # Bright Green or Red
        sign = "+" if alpha > 0 else ""
        kpis["Alpha (vs B&H)"] = f"<span style='color:{color}; font-weight:bold'>{sign}{alpha:.2%}</span>"

    # Initial Capital
    initial_cap = summary_dict.get('financial/initial_capital')
    if initial_cap is not None:
        kpis["Initial Capital"] = f"${initial_cap:,.2f}"

    # Final Net Worth
    final_nw = summary_dict.get('financial/final_networth')
    if final_nw is not None:
        kpis["Final Net Worth"] = f"${final_nw:,.2f}"

    return kpis

def create_html_report(metrics_df, summary_dict):
    print("📊 Generating Professional Quant Report...")
    
    # 1. KPIs Section
    kpis = calculate_financial_kpis(metrics_df, summary_dict)
    
    # 2. Interactive Chart: Training Reward Over Time
    # We look for ANY of these columns to plot the main line
    potential_cols = [
        'best_eval/mean_reward',  # Best metric (Validation)
        'rollout/ep_rew_mean',    # Second best (Training)
        'train/reward',           # Alternative
        'mean_reward'             # Generic
    ]
    
    # Find the first column that actually exists in the DF
    plot_col = next((c for c in potential_cols if c in metrics_df.columns), None)

    if not plot_col:
        print(f"⚠️ Warning: No reward columns found. Available: {list(metrics_df.columns)}")
        chart_html = "<p style='color:red; text-align:center;'>Reward data not available in history.</p>"
        chart_title = "Data Missing"
    else:
        print(f"   > Using '{plot_col}' for main chart.")
        chart_title = f"Performance: {plot_col}"
        
        # Filter NaNs
        # Use global_step for X axis if available, else index
        x_col = 'global_step' if 'global_step' in metrics_df.columns else None
        
        if x_col:
            plot_df = metrics_df[[x_col, plot_col]].dropna()
            x_data = plot_df[x_col]
        else:
            plot_df = metrics_df[[plot_col]].dropna()
            x_data = plot_df.index

        y_data = plot_df[plot_col]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data, 
            y=y_data, 
            mode='lines', 
            name='Reward',
            line=dict(color='#00ff00', width=2)
        ))
        fig.update_layout(
            title=chart_title,
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Steps",
            yaxis_title="Reward"
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # --- HTML GENERATION ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Bot - Quant Report</title>
        <style>
            body {{ background-color: #1e1e1e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
            .container {{ max_width: 1200px; margin: auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 20px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .kpi-card {{ background: #2d2d2d; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            .kpi-value {{ font-size: 24px; font-weight: bold; color: #fff; margin-top: 10px; }}
            .kpi-label {{ font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }}
            .section {{ background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            h2 {{ border-bottom: 2px solid #444; padding-bottom: 10px; margin-top: 0; color: #00ff00; }}
            .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }}
            img {{ max-width: 100%; border-radius: 5px; border: 1px solid #444; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
            td, th {{ padding: 12px; border-bottom: 1px solid #444; text-align: left; }}
            tr:hover {{ background-color: #383838; }}
            .footer {{ text-align: center; color: #666; margin-top: 40px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>🤖 AI Trading Bot Report</h1>
                    <span style="color: #888;">Run ID: {summary_dict.get('run_id', 'N/A')} | {summary_dict.get('_timestamp', '')}</span>
                </div>
            </div>

            <!-- Dynamic KPI Grid -->
            <div class="kpi-grid">
    """
    
    # Loop through all calculated KPIs and create cards
    for key, val in kpis.items():
        html_content += f"""
                <div class="kpi-card">
                    <div class="kpi-label">{key}</div>
                    <div class="kpi-value">{val}</div>
                </div>
        """

    html_content += f"""
            </div>

            <div class="section">
                <h2>📈 Training Performance</h2>
                {chart_html}
            </div>

            <div class="section">
                <h2>⚙️ System Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    # Filter summary dict for interesting config values (exclude large arrays/objects)
    exclude_keys = ['_wandb', 'graph', 'code', 'media']
    for key, val in sorted(summary_dict.items()):
        if key not in exclude_keys and isinstance(val, (int, float, str, bool)):
            html_content += f"<tr><td>{key}</td><td>{val}</td></tr>"

    html_content += """
                </table>
            </div>
            
            <div class="footer">
                Generated by fetch_metrics.py
            </div>
        </div>
    </body>
    </html>
    """

    with open("results/quant_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ Successfully generated: results/quant_report.html")

    
def _generate_metrics_worker():
    api = wandb.Api()
    # Ensure this matches your project
    runs = api.runs("zhivko/ai-trading-bot")

    if runs:
        run = runs[0]
        print(f"Fetching metrics for run: {run.name} ({run.id})")
        os.makedirs("results", exist_ok=True)

        # 1. Fetch History (More samples to catch sparse eval metrics)
        print("Downloading history...")
        # Increase samples to ensure we capture the 'best_eval' points
        history_df = run.history(samples=10000)
        history_df.to_csv("results/metrics.csv")

        # 2. Fetch Summary
        summary_dict = run.summary._json_dict
        pd.DataFrame(list(summary_dict.items()), columns=["key", "value"]).to_csv("results/summary.csv", index=False)

        # 3. Generate Report
        create_html_report(history_df, summary_dict)

        # 4. Git Push
        try:
            print("Pushing to Git...")
            subprocess.run(["git", "add", "results/"], check=True)
            subprocess.run(["git", "commit", "-m", f"Report update {run.name}"], check=True)
            subprocess.run(["git", "push"], check=True)
        except subprocess.CalledProcessError as e:
             if e.returncode == 1: print("Nothing to commit.")
             else: print(f"Git error: {e}")
    else:
        print("No runs found.")

def generate_metrics():
    thread = threading.Thread(target=_generate_metrics_worker)
    thread.start()

if __name__ == "__main__":
    generate_metrics()