import wandb
import subprocess
import os
import shutil
import time
import threading
import pandas as pd
import base64

def generate_metrics():
    def _worker():
        api = wandb.Api()

        # Get the latest run from the project
        runs = api.runs("zhivko/ai-trading-bot")
        if runs:
            run = runs[0]  # Latest run
            print(f"Fetching metrics for run: {run.name} ({run.id})")

            # Delete all files in results directory if it exists
            if os.path.exists("results"):
                for file in os.listdir("results"):
                    file_path = os.path.join("results", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)

            # save the metrics for the run to a csv file
            run.history().to_csv("results/metrics.csv")
            print("Metrics saved to results/metrics.csv")

            summary_df = pd.DataFrame(list(run.summary.items()), columns=["key", "value"])
            summary_df.to_csv("results/summary.csv", index=False)
            print("Summary saved to results/summary.csv")

            # Download and save plots/images from the run to results directory
            files = run.files()
            image_files = [f for f in files if hasattr(f, 'mimetype') and f.mimetype and f.mimetype.startswith('image/')]
            downloaded_images = []

            for img_file in image_files:
                try:
                    # 1. Download file to current directory (no root to avoid subdir issues)
                    downloaded_filename = img_file.name
                    img_file.download(replace=True)

                    # 2. Wait for file handle release
                    time.sleep(0.3)

                    # 3. Parse step from filename
                    parts = downloaded_filename.split('_')
                    step = ''
                    if 'thread_0_chart' in downloaded_filename:
                        step_part = parts[-2] if len(parts) > 2 else ''
                        step = step_part.replace('thread_0_chart', '').strip('-')
                    elif 'feature_importance' in downloaded_filename:
                        step_part = parts[1] if len(parts) > 1 else ''
                        step = step_part.replace('feature_importance', '').strip('-')

                    # 4. New filename with run ID and step
                    base_filename = os.path.basename(downloaded_filename)
                    new_filename = f"run_{run.id}_step_{step}_{base_filename}" if step else f"run_{run.id}_{base_filename}"
                    destination_path = os.path.join("results", new_filename)

                    # 5. Move to results/
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            os.rename(downloaded_filename, destination_path)
                            print(f"Downloaded and saved: {new_filename}")
                            downloaded_images.append(new_filename)
                            break
                        except PermissionError:
                            if attempt < max_retries - 1:
                                time.sleep(1)
                            else:
                                print(f"Failed to move {downloaded_filename} after {max_retries} attempts: {e}")

                except Exception as e:
                    print(f"Failed to download {img_file.name}: {e}")

            # Generate HTML report
            metrics_df = run.history()
            html_content = f"""
            <html>
            <head><title>Trading Bot Metrics Report - Run {run.id}</title></head>
            <body>
            <h1>Trading Bot Metrics Report</h1>
            <h2>Run: {run.name} ({run.id})</h2>
            <h2>Summary</h2>
            {summary_df.to_html()}
            <h2>Metrics</h2>
            {metrics_df.to_html()}
            <h2>Charts</h2>
            """
            for img in downloaded_images:
                img_path = os.path.join("results", img)
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                ext = os.path.splitext(img)[1].lower()
                mime = 'image/png' if ext == '.png' else 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                html_content += f'<img src="data:{mime};base64,{img_data}" alt="{img}" style="max-width:100%;"><br>'
            html_content += "</body></html>"
            with open("results/report.html", "w") as f:
                f.write(html_content)
            print("Report generated: results/report.html")

            # Cleanup empty 'media' folders
            media_dir = os.path.join("results", "media")
            if os.path.exists(media_dir):
                try:
                    shutil.rmtree(media_dir)
                except Exception:
                    pass

            # Delete all downloaded images
            for img in downloaded_images:
                img_path = os.path.join("results", img)
                try:
                    os.remove(img_path)
                    print(f"Deleted image: {img}")
                except Exception as e:
                    print(f"Failed to delete {img}: {e}")

            # Commit and push
            try:
                subprocess.run(["git", "add", "results/"], check=True)
                subprocess.run(["git", "commit", "-m", f"Update metrics for run {run.name}"], check=True)
                subprocess.run(["git", "push"], check=True)
                print("results/ directory committed and pushed to remote")
            except subprocess.CalledProcessError as e:
                print(f"Git operation failed: {e}")
        else:
            print("No runs found in the project")

    thread = threading.Thread(target=_worker)
    thread.start()

if __name__ == "__main__":
    generate_metrics()