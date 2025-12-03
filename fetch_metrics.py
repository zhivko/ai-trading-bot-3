import wandb
import subprocess
import os
import shutil
import time
import pandas as pd
import plotly.graph_objects as go
import glob
import base64
import threading

def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def calculate_financial_kpis(history_df, summary_dict):
    """
    Derive financial metrics using Summary (for latest) and History (for trends).
    """
    kpis = {}
    
    # --- 1. Final Mean Reward ---
    # Priority: Summary -> Best Eval -> Rollout Reward
    if 'best_eval/mean_reward' in summary_dict:
        val = summary_dict['best_eval/mean_reward']
        kpis["Final Eval Reward"] = f"{val:.2f}"
    elif 'rollout/ep_rew_mean' in summary_dict:
        val = summary_dict['rollout/ep_rew_mean']
        kpis["Final Mean Reward"] = f"{val:.2f}"
    # Fallback to History
    elif 'best_eval/mean_reward' in history_df.columns:
        kpis["Final Eval Reward"] = f"{history_df['best_eval/mean_reward'].dropna().iloc[-1]:.2f}"
    elif 'rollout/ep_rew_mean' in history_df.columns:
        kpis["Final Mean Reward"] = f"{history_df['rollout/ep_rew_mean'].dropna().iloc[-1]:.2f}"
    else:
        kpis["Final Reward"] = "N/A"

    # --- 2. Total Steps ---
    if 'global_step' in summary_dict:
        total = summary_dict['global_step']
    elif 'global_step' in history_df.columns:
        total = history_df['global_step'].max()
    else:
        total = len(history_df)
    kpis["Total Steps"] = f"{total:,.0f}"

    # --- 3. Training Time ---
    # Check multiple keys for time
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
        kpis["Est. Training Time"] = f"{float(time_val) / 3600:.1f} Hours"
    else:
        kpis["Est. Training Time"] = "N/A"

    return kpis

def create_html_report(metrics_df, summary_dict):
    print("📊 Generating Professional Quant Report...")
    
    # 1. KPIs Section
    kpis = calculate_financial_kpis(metrics_df, summary_dict)
    
    # 2. Interactive Chart: Robust Column Selection
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
        plot_df = metrics_df[['global_step', plot_col]].dropna() if 'global_step' in metrics_df.columns else metrics_df[[plot_col]].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df['global_step'] if 'global_step' in plot_df.columns else plot_df.index, 
            y=plot_df[plot_col], 
            mode='lines', 
            name='Reward',
            line=dict(color='#00ff00', width=2)
        ))
        fig.update_layout(
            title=chart_title,
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # 3. Find Images
    feature_imp_files = glob.glob("results/feature_importance*.png")
    trade_chart_files = glob.glob("results/trade_analysis*.png") # Renamed files

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
            .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .kpi-card {{ background: #2d2d2d; padding: 20px; border-radius: 8px; text-align: center; }}
            .kpi-value {{ font-size: 24px; font-weight: bold; color: #00ff00; }}
            .kpi-label {{ font-size: 14px; color: #888; }}
            .section {{ background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            h2 {{ border-bottom: 2px solid #444; padding-bottom: 10px; margin-top: 0; }}
            .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            img {{ max-width: 100%; border-radius: 5px; border: 1px solid #444; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ padding: 10px; border-bottom: 1px solid #444; text-align: left; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Trading Bot Analysis Report</h1>
                <p>Generated by fetch_metrics.py</p>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Reward Metric</div>
                    <div class="kpi-value">{kpis.get('Final Eval Reward', kpis.get('Final Mean Reward', 'N/A'))}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Training Steps</div>
                    <div class="kpi-value">{kpis.get('Total Steps', 'N/A')}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Training Duration</div>
                    <div class="kpi-value">{kpis.get('Est. Training Time', 'N/A')}</div>
                </div>
            </div>

            <div class="section">
                <h2>📈 Training Progress</h2>
                {chart_html}
            </div>

            <div class="section">
                <h2>🧠 Feature Importance</h2>
                <div class="img-grid">
    """
    
    # Sort feature importance by date/number usually embedded in filename
    for f in sorted(feature_imp_files):
        b64 = get_base64_image(f)
        html_content += f'<div><img src="data:image/png;base64,{b64}" /></div>'

    html_content += """
                </div>
            </div>

            <div class="section">
                <h2>🐂🐻 Regime Analysis (Bull vs Bear)</h2>
                <div class="img-grid">
    """
    # Sort and show last 4 trade charts
    trade_chart_files.sort(key=os.path.getmtime)
    for f in trade_chart_files[-4:]:
        b64 = get_base64_image(f)
        fname = os.path.basename(f)
        html_content += f'<div><p>{fname}</p><img src="data:image/png;base64,{b64}" /></div>'

    html_content += """
                </div>
            </div>

            <div class="section">
                <h2>⚙️ Configuration (Hyperparameters)</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
    """
    # Add Summary Table
    for key, val in summary_dict.items():
        if isinstance(val, (int, float, str)): # Skip complex objects
            html_content += f"<tr><td>{key}</td><td>{val}</td></tr>"

    html_content += """
                </table>
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

        # 3. Download Images (Robust)
        files = run.files()
        image_files = [f for f in files if hasattr(f, 'mimetype') and f.mimetype and f.mimetype.startswith('image/')]

        for img_file in image_files:
            try:
                img_file.download(root="results", replace=True)
                time.sleep(0.2)

                filename = os.path.basename(img_file.name)
                downloaded_path = os.path.normpath(os.path.join("results", img_file.name))
                destination_path = os.path.join("results", filename)

                if os.path.exists(downloaded_path):
                    if os.path.exists(destination_path):
                        try:
                            os.remove(destination_path)
                        except PermissionError:
                            time.sleep(1)
                            try: os.remove(destination_path)
                            except: continue
                    try:
                        os.rename(downloaded_path, destination_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to download {img_file.name}: {e}")

        # Cleanup media
        if os.path.exists(os.path.join("results", "media")):
            try: shutil.rmtree(os.path.join("results", "media"))
            except: pass

        # 4. Generate Report
        create_html_report(history_df, summary_dict)

        # 5. Remove images before committing (since they are embedded in HTML)
        image_files_to_remove = glob.glob("results/*.png")
        for img in image_files_to_remove:
            try:
                os.remove(img)
                print(f"Removed image: {img}")
            except Exception as e:
                print(f"Failed to remove {img}: {e}")

        # 6. Git Push
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