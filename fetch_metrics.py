import wandb
import subprocess
import os
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import logging
import numpy as np

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

def create_comprehensive_chart(metrics_df, summary_dict):
    """
    Create a comprehensive multi-subplot chart with actions, networth, reward components, and total reward.
    Similar to the _plot_regime_chart method in callbacks/base_callbacks.py
    """
    logging.info("Creating comprehensive trading analysis chart...")
    
    # Check what data columns are available
    available_columns = list(metrics_df.columns)
    logging.info(f"Available columns: {available_columns}")
    
    # X-axis data
    x_col = 'global_step' if 'global_step' in metrics_df.columns else metrics_df.index
    steps = np.arange(len(metrics_df))
    
    # Determine what subplots to create based on available data
    subplot_count = 1  # Base reward subplot
    has_actions = any(col in available_columns for col in ['action', 'actions', 'train/action'])
    has_networth = any(col in available_columns for col in ['net_worth', 'portfolio_value', 'financial/final_networth'])
    has_reward_components = any(col in available_columns for col in [
        'reward_base', 'reward_fee', 'reward_action_change', 'reward_trend', 
        'reward_holding', 'reward_inertia', 'reward_closer', 'reward_overtrade', 'reward_episode'
    ])
    has_price_data = any(col in available_columns for col in ['current_price', 'price', 'price/current_price'])
    has_ema_data = any(col in available_columns for col in ['ema50', 'price/ema50'])
    
    # Count additional subplots
    if has_price_data: subplot_count += 1  # Price/EMA subplot
    if has_actions: subplot_count += 1     # Actions subplot
    if has_networth: subplot_count += 1    # Networth subplot
    if has_reward_components: subplot_count += 2  # Components + Total reward subplots
    
    # Create subplot titles
    subplot_titles = []
    if has_price_data:
        subplot_titles.append("Price & EMA")
    subplot_titles.append("Training Performance")
    if has_actions:
        subplot_titles.append("Actions")
    if has_networth:
        subplot_titles.append("Net Worth")
    if has_reward_components:
        subplot_titles.append("Reward Components")
        subplot_titles.append("Total Reward")
    
    # Create subplots
    fig = make_subplots(
        rows=subplot_count, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False}] for _ in range(subplot_count)]
    )
    
    current_row = 1
    
    # 1. Price & EMA subplot
    if has_price_data:
        price_col = next((col for col in ['current_price', 'price', 'price/current_price'] if col in available_columns), None)
        if price_col:
            prices = metrics_df[price_col].fillna(method='ffill').fillna(method='bfill')
            fig.add_trace(
                go.Scatter(x=steps, y=prices, name='Price', line=dict(color='black', width=1.2)),
                row=current_row, col=1
            )
            
            # Add EMA if available
            if has_ema_data:
                ema_col = next((col for col in ['ema50', 'price/ema50'] if col in available_columns), None)
                if ema_col:
                    emas = metrics_df[ema_col].fillna(method='ffill').fillna(method='bfill')
                    fig.add_trace(
                        go.Scatter(x=steps, y=emas, name='EMA 50', line=dict(color='orange', width=1, dash='dash')),
                        row=current_row, col=1
                    )
        
        current_row += 1
    
    # 2. Main reward subplot (training performance)
    potential_reward_cols = [
        'best_eval/mean_reward', 'rollout/ep_rew_mean', 'train/reward', 'mean_reward',
        'eval/mean_reward', 'reward'
    ]
    reward_col = next((col for col in potential_reward_cols if col in available_columns), None)
    
    if reward_col:
        rewards = metrics_df[reward_col].fillna(method='ffill').fillna(method='bfill')
        fig.add_trace(
            go.Scatter(x=steps, y=rewards, name=f'{reward_col}', line=dict(color='#00ff00', width=2)),
            row=current_row, col=1
        )
        chart_title = f"Trading Performance: {reward_col}"
    else:
        chart_title = "Trading Performance: No reward data available"
        logging.warning("No reward columns found for main chart")
    
    current_row += 1
    
    # 3. Actions subplot
    if has_actions:
        action_col = next((col for col in ['action', 'actions', 'train/action'] if col in available_columns), None)
        if action_col:
            actions = metrics_df[action_col].fillna(0)
            colors = ['green' if a >= 0 else 'red' for a in actions]
            fig.add_trace(
                go.Bar(x=steps, y=actions, name='Actions', marker_color=colors, opacity=0.7),
                row=current_row, col=1
            )
        current_row += 1
    
    # 4. Networth subplot
    if has_networth:
        networth_col = next((col for col in ['net_worth', 'portfolio_value', 'financial/final_networth'] if col in available_columns), None)
        if networth_col:
            networth = metrics_df[networth_col].fillna(method='ffill').fillna(method='bfill')
            fig.add_trace(
                go.Scatter(x=steps, y=networth, name='Net Worth', line=dict(color='blue', width=1.5)),
                row=current_row, col=1
            )
        current_row += 1
    
    # 5. Reward components subplot
    if has_reward_components:
        reward_component_mapping = {
            'reward_base': 'Base (net worth)',
            'reward_fee': 'Fee penalty',
            'reward_action_change': 'Action change penalty',
            'reward_trend': 'Trend alignment',
            'reward_holding': 'Holding cost',
            'reward_inertia': 'Inertia penalty',
            'reward_closer': 'Closer bonus',
            'reward_overtrade': 'Overtrading penalty',
            'reward_episode': 'Episode termination'
        }
        
        # Color palette for components
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        
        component_count = 0
        for col, label in reward_component_mapping.items():
            if col in available_columns:
                component_data = metrics_df[col].fillna(0)
                fig.add_trace(
                    go.Scatter(x=steps, y=component_data, name=label, 
                             line=dict(color=colors[component_count % len(colors)], width=1, dash='solid')),
                    row=current_row, col=1
                )
                component_count += 1
        
        current_row += 1
        
        # 6. Total reward subplot (sum of all components)
        if component_count > 0:
            # Calculate total reward as sum of available components
            total_reward = np.zeros(len(metrics_df))
            for col in reward_component_mapping.keys():
                if col in available_columns:
                    total_reward += metrics_df[col].fillna(0).values
            
            fig.add_trace(
                go.Scatter(x=steps, y=total_reward, name='Total Reward', 
                         line=dict(color='black', width=2)),
                row=current_row, col=1
            )
            # Add horizontal line at zero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(text=chart_title, x=0.5, font=dict(size=16)),
        template="plotly_dark",
        height=300 * subplot_count,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Steps", row=subplot_count, col=1)
    
    # Update y-axis labels for each subplot
    if has_price_data:
        fig.update_yaxes(title_text="Price", row=1, col=1)
    
    reward_row = 2 if has_price_data else 1
    fig.update_yaxes(title_text="Reward", row=reward_row, col=1)
    
    if has_actions:
        action_row = reward_row + 1
        fig.update_yaxes(title_text="Actions", row=action_row, col=1)
    
    if has_networth:
        networth_row = action_row + 1 if has_actions else reward_row + 1
        fig.update_yaxes(title_text="Net Worth", row=networth_row, col=1)
    
    if has_reward_components:
        components_row = networth_row + 1 if has_networth else (action_row + 1 if has_actions else reward_row + 1)
        fig.update_yaxes(title_text="Reward Components", row=components_row, col=1)
        
        total_row = components_row + 1
        fig.update_yaxes(title_text="Total Reward", row=total_row, col=1)
    
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    logging.info(f"✅ Created comprehensive chart with {subplot_count} subplots")
    return chart_html, chart_title

def create_html_report(metrics_df, summary_dict):
    logging.info("📊 Generating Professional Quant Report...")

    # Get Git information
    git_info = {}
    try:
        git_info['local_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
        git_info['commit_id'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        try:
            remote_branch = subprocess.check_output(['git', 'rev-parse', '--symbolic-full-name', '@{u}']).decode('utf-8').strip()
            git_info['remote_branch'] = remote_branch
        except subprocess.CalledProcessError:
            git_info['remote_branch'] = 'No upstream'
    except subprocess.CalledProcessError:
        git_info['local_branch'] = 'N/A'
        git_info['commit_id'] = 'N/A'
        git_info['remote_branch'] = 'N/A'

    # 1. KPIs Section
    kpis = calculate_financial_kpis(metrics_df, summary_dict)
    
    # 2. Enhanced Interactive Chart: Comprehensive Trading Analysis
    chart_html, chart_title = create_comprehensive_chart(metrics_df, summary_dict)

    # --- HTML GENERATION ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Bot - Quant Report</title>
        <style>
            body {{ background-color: #1e1e1e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: auto; }}
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
                    <br><span style="color: #666; font-size: 12px;">Git: Branch {git_info['local_branch']} ({git_info['remote_branch']}) | Commit {git_info['commit_id'][:8]}</span>
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
                <h2>📊 Visualization</h2>
                <div class="img-grid">
                    <img src="trading_performance.png" alt="Trading Performance">
                    <img src="average_saliency.png" alt="Average Saliency">
                </div>
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
    logging.info("✅ Successfully generated: results/quant_report.html")

    
def _generate_metrics_worker():
    api = wandb.Api()
    # Ensure this matches your project
    runs = api.runs("zhivko/ai-trading-bot")

    if runs:
        run = runs[0]
        logging.info(f"Fetching metrics for run: {run.name} ({run.id})")
        os.makedirs("results", exist_ok=True)

        # 1. Fetch History (More samples to catch sparse eval metrics)
        logging.info("Downloading history...")
        # Increase samples to ensure we capture the 'best_eval' points
        history_df = run.history(samples=10000)
        history_df.to_csv("results/metrics.csv")

        # 2. Fetch Summary
        summary_dict = run.summary._json_dict
        pd.DataFrame(list(summary_dict.items()), columns=["key", "value"]).to_csv("results/summary.csv", index=False)

        # 3. Generate Report
        create_html_report(history_df, summary_dict)

        # 4. Git Push
        '''
        try:
            logging.info("Pushing to Git...")
            subprocess.run(["git", "add", "results/"], check=True)
            subprocess.run(["git", "commit", "-m", f"Report update {run.name}"], check=True)
            subprocess.run(["git", "push"], check=True)
        except subprocess.CalledProcessError as e:
             if e.returncode == 1: logging.info("Nothing to commit.")
             else: logging.info(f"Git error: {e}")
        '''
    else:
        logging.info("No runs found.")

def generate_metrics():
    thread = threading.Thread(target=_generate_metrics_worker)
    thread.start()

if __name__ == "__main__":
    generate_metrics()