"""
Neural Network Visualization Demo for AI Trading Bot
===================================================

This script demonstrates how to use the NeuralNetworkVisualizer with the
actual RecurrentPPO model from the AI trading bot project.

Integration with existing codebase:
- Uses the same feature extraction from features.py
- Compatible with existing callbacks
- Works with the trading environment structure
- Supports the 220-feature input format

Author: AI Trading Bot Team
Date: 2025-12-13
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpLstmPolicy
import warnings

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network_visualizer import NeuralNetworkVisualizer
from features import get_features
from enhanced_trading_env import EnhancedTradingEnv


def load_model_and_data():
    """
    Load the trained model and sample data for visualization
    
    Returns:
        tuple: (model, observations, feature_names)
    """
    
    # Try to load the trained model
    model_path = "models/recurrent_ppo_trading_model.zip"  # Adjust path as needed
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            model = PPO.load(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model for demonstration...")
            model = create_demo_model()
    else:
        print("Model not found, creating a new model for demonstration...")
        model = create_demo_model()
    
    # Create sample observations using the same feature extraction
    observations, feature_names = create_sample_observations_with_features()
    
    return model, observations, feature_names


def create_demo_model():
    """Create a demo RecurrentPPO model"""
    
    # Create a simple environment for model initialization
    env = EnhancedTradingEnv()
    
    # Create RecurrentPPO model with LSTM policy
    model = PPO(
        MlpLstmPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=2,
            net_arch=dict(pi=[512, 512], vf=[512, 512])
        )
    )
    
    # Do a quick dummy training step to initialize weights
    print("Initializing model weights...")
    try:
        model.learn(total_timesteps=100)
        print("✓ Model initialized")
    except Exception as e:
        print(f"Warning during model initialization: {e}")
        print("Continuing with uninitialized model...")
    
    return model


def create_sample_observations_with_features(n_samples=50):
    """
    Create sample observations using the same feature extraction logic
    """
    print("Creating sample observations with real feature extraction...")
    
    # Load sample data
    try:
        # Try to load real trading data
        df = pd.read_csv('BTCUSDT_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Load volume profile data (create mock if not available)
        vp7_df = create_mock_volume_profile_data(df)
        vp30_df = create_mock_volume_profile_data(df)
        
        # Extract features for multiple timestamps
        observations = []
        feature_names = []
        
        # Create feature names (this should match the features.py logic)
        feature_names = create_feature_names()
        
        # Sample timestamps for feature extraction
        sample_timestamps = df.index[-100:]  # Last 100 data points
        
        for i, t in enumerate(sample_timestamps[-n_samples:]):
            try:
                features = get_features(df, vp7_df, vp30_df, t)
                observations.append(features)
                
                if i == 0:  # Print first feature vector info
                    print(f"✓ Feature vector shape: {features.shape}")
                    print(f"✓ Total features: {len(feature_names)}")
                    
            except Exception as e:
                print(f"Warning: Could not extract features for timestamp {t}: {e}")
                # Use zero features as fallback
                observations.append(np.zeros(220))
        
        observations = np.array(observations)
        print(f"✓ Created {len(observations)} sample observations")
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating synthetic observations...")
        
        # Create synthetic observations
        observations = create_synthetic_observations(n_samples)
        feature_names = create_feature_names()
    
    return observations, feature_names


def create_mock_volume_profile_data(df):
    """Create mock volume profile data"""
    mock_data = {}
    
    for timestamp in df.index:
        mock_data[timestamp] = {
            'poc': df.loc[timestamp, 'close'],
            'vah': df.loc[timestamp, 'close'] * 1.02,
            'val': df.loc[timestamp, 'close'] * 0.98,
            'hvn': [df.loc[timestamp, 'close'] * 0.99, df.loc[timestamp, 'close'] * 1.01],
            'lvn': [df.loc[timestamp, 'close'] * 0.97, df.loc[timestamp, 'close'] * 1.03],
            'heatmap': np.random.exponential(1, 40)
        }
    
    return pd.DataFrame(mock_data).T


def create_synthetic_observations(n_samples):
    """Create synthetic observations when real data is not available"""
    np.random.seed(42)
    
    observations = []
    for i in range(n_samples):
        # Create realistic synthetic features
        obs = np.zeros(220)
        
        # Price data (first few features)
        base_price = 50000
        obs[0] = base_price + np.random.normal(0, 1000)
        
        # Volume profile heatmaps (features 0-79)
        obs[0:40] = np.random.exponential(1, 40)  # 7-day VP
        obs[40:80] = np.random.exponential(1, 40)  # 30-day VP
        
        # Normalized VP levels (features 80-89)
        obs[80:83] = np.random.uniform(-0.1, 0.1, 3)  # 7-day levels
        obs[83:86] = np.random.uniform(-0.1, 0.1, 3)  # 30-day levels
        
        # VP statistics (features 86-97)
        obs[86:92] = np.random.uniform(0, 5, 6)  # 7-day stats
        obs[92:98] = np.random.uniform(0, 5, 6)  # 30-day stats
        
        # Distance and relative features (features 98-105)
        obs[98:105] = np.random.uniform(-0.1, 0.1, 7)
        
        # Technical indicators (features 105-113)
        obs[105:108] = np.random.uniform(-1, 1, 3)  # MACD
        obs[108] = np.random.uniform(0, 100)  # RSI
        obs[109:111] = np.random.uniform(0, 1, 2)  # Stoch RSI
        obs[111] = np.random.uniform(0, 1)  # ATR
        obs[112] = np.random.uniform(-0.1, 0.1)  # EMA normalized
        
        # Session encoding (features 113-136)
        hour = i % 24
        obs[113 + hour] = 1.0
        
        observations.append(obs)
    
    return np.array(observations)


def create_feature_names():
    """Create feature names matching the features.py logic"""
    feature_names = []
    
    # Volume profile heatmaps
    feature_names.extend([f"VP_7d_Bin_{i}" for i in range(40)])
    feature_names.extend([f"VP_30d_Bin_{i}" for i in range(40)])
    
    # Normalized VP levels
    feature_names.extend(["VP_7d_POC_norm", "VP_7d_VAH_norm", "VP_7d_VAL_norm"])
    feature_names.extend(["VP_30d_POC_norm", "VP_30d_VAH_norm", "VP_30d_VAL_norm"])
    
    # VP statistics
    feature_names.extend([
        "VP_7d_HVN_count", "VP_7d_HVN_avg_dist", "VP_7d_HVN_nearest",
        "VP_7d_LVN_count", "VP_7d_LVN_avg_dist", "VP_7d_LVN_nearest"
    ])
    feature_names.extend([
        "VP_30d_HVN_count", "VP_30d_HVN_avg_dist", "VP_30d_HVN_nearest",
        "VP_30d_LVN_count", "VP_30d_LVN_avg_dist", "VP_30d_LVN_nearest"
    ])
    
    # Distance and relative features
    feature_names.extend([
        "VP_7d_dist_HVN", "VP_7d_dist_LVN", "VP_7d_rel_POC", "VP_7d_in_VA",
        "VP_30d_dist_HVN", "VP_30d_dist_LVN", "VP_30d_rel_POC", "VP_30d_in_VA",
        "volatility", "orderbook_imbalance"
    ])
    
    # Technical indicators
    feature_names.extend([
        "MACD_line", "MACD_signal", "MACD_hist", "RSI", "Stoch_K", "Stoch_D", "ATR", "EMA_50_norm"
    ])
    
    # Session encoding
    feature_names.extend([f"session_hour_{i}" for i in range(24)])
    
    return feature_names


def run_visualization_demo():
    """Run the complete visualization demonstration"""
    
    print("🧠 Neural Network Visualization Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("neural_network_analysis")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load model and data
        model, observations, feature_names = load_model_and_data()
        
        # Create visualizer
        print("\n📊 Creating Neural Network Visualizer...")
        visualizer = NeuralNetworkVisualizer(
            model=model,
            feature_names=feature_names,
            save_dir=str(output_dir)
        )
        print("✓ Visualizer created")
        
        # Run individual visualizations
        print("\n🎯 Generating Visualizations...")
        
        # 1. Model Architecture
        print("  1. Model Architecture...")
        arch_fig = visualizer.visualize_model_architecture(
            save_path=str(output_dir / "01_model_architecture.html")
        )
        
        # 2. Layer Weights Analysis
        print("  2. Layer Weights Analysis...")
        weights_fig = visualizer.analyze_layer_weights(
            save_path=str(output_dir / "02_layer_weights.html")
        )
        
        # 3. LSTM Hidden States (if we have observations)
        if observations is not None and len(observations) > 0:
            print("  3. LSTM Hidden States...")
            lstm_fig = visualizer.track_lstm_hidden_states(
                observations,
                save_path=str(output_dir / "03_lstm_hidden_states.html")
            )
            
            # 4. Activation Patterns
            print("  4. Activation Patterns...")
            activation_fig = visualizer.visualize_activation_patterns(
                observations,
                save_path=str(output_dir / "04_activation_patterns.html")
            )
        
        # 5. Feature Importance Evolution (simulated)
        print("  5. Feature Importance Evolution...")
        training_steps = [1000, 5000, 10000, 20000, 50000]
        importance_scores = []
        
        # Simulate feature importance evolution
        for step in training_steps:
            # Create evolving importance scores
            base_importance = np.random.uniform(0.1, 1.0, len(feature_names))
            # Add some temporal evolution
            evolution_factor = 1.0 + 0.5 * np.sin(step / 10000)
            importance_scores.append(base_importance * evolution_factor)
        
        importance_fig = visualizer.visualize_feature_importance_evolution(
            training_steps,
            importance_scores,
            save_path=str(output_dir / "05_feature_importance_evolution.html")
        )
        
        # 6. Decision Flow (if Captum is available)
        try:
            if observations is not None and len(observations) > 0:
                print("  6. Decision Flow...")
                decision_fig = visualizer.visualize_decision_flow(
                    observations[0],  # Single observation
                    save_path=str(output_dir / "06_decision_flow.html")
                )
        except ImportError:
            print("  6. Decision Flow... (Skipped - Captum not available)")
        except Exception as e:
            print(f"  6. Decision Flow... (Error: {e})")
        
        # 7. Create comprehensive report
        print("  7. Comprehensive Report...")
        report_files = visualizer.create_comprehensive_report(
            observations=observations,
            save_dir=str(output_dir)
        )
        
        # Print results
        print("\n✅ Visualization Complete!")
        print(f"📁 Output directory: {output_dir}")
        print("\nGenerated files:")
        for name, path in report_files.items():
            print(f"  • {name}: {os.path.basename(path)}")
        
        # Create a simple index file
        create_index_file(output_dir, report_files)
        
        print(f"\n🌐 Open {output_dir}/00_summary_report.html in your browser to view all visualizations")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def create_index_file(output_dir: Path, report_files: dict):
    """Create a simple index file for easy navigation"""
    
    index_path = output_dir / "index.html"
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Analysis - Index</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #ecf0f1; }
        .link { color: #3498db; text-decoration: none; font-weight: bold; }
        .link:hover { text-decoration: underline; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { border: 1px solid #bdc3c7; padding: 20px; border-radius: 8px; background: white; }
        .card h3 { margin-top: 0; color: #2c3e50; }
        .summary { background: #e8f6f3; padding: 20px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Neural Network Analysis Report</h1>
        
        <div class="summary">
            <h2>📊 Analysis Summary</h2>
            <p>This report provides comprehensive visualizations of the RecurrentPPO model used in the AI trading bot.</p>
            <p>The analysis covers model architecture, layer weights, LSTM dynamics, and activation patterns to help understand how the model makes trading decisions.</p>
        </div>
        
        <div class="grid">
"""
    
    # Add sections for each report
    section_info = {
        'summary': ('📋 Summary Report', 'Overview of all visualizations and key insights'),
        'model_architecture': ('🏗️ Model Architecture', 'Complete network structure with LSTM layers and MLP heads'),
        'layer_weights': ('⚖️ Layer Weights Analysis', 'Weight distributions and statistics for each layer'),
        'lstm_hidden_states': ('🔄 LSTM Hidden States', 'Hidden state evolution tracking during trading sequences'),
        'activation_patterns': ('⚡ Activation Patterns', 'Neuron activations for different market conditions'),
    }
    
    for key, (title, description) in section_info.items():
        if key in report_files:
            filename = os.path.basename(report_files[key])
            html_content += f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{description}</p>
                <a href="{filename}" class="link" target="_blank">View Analysis →</a>
            </div>
            """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>🔍 How to Use This Report</h2>
            <ul>
                <li><strong>Model Architecture:</strong> Understand the network design and component relationships</li>
                <li><strong>Layer Weights:</strong> Monitor weight distributions to detect overfitting or training issues</li>
                <li><strong>LSTM Hidden States:</strong> Track how the model maintains memory across time steps</li>
                <li><strong>Activation Patterns:</strong> Analyze neural activity to debug unexpected behavior</li>
                <li><strong>Feature Importance:</strong> See which inputs the model considers most important</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🎯 Key Insights for Trading</h2>
            <ul>
                <li><strong>Temporal Learning:</strong> LSTM layers capture market memory and trends</li>
                <li><strong>Feature Processing:</strong> 220 features including price, volume, and technical indicators</li>
                <li><strong>Decision Making:</strong> Actor-critic architecture separates action and value estimation</li>
                <li><strong>Market Adaptation:</strong> Network learns to adapt to different market conditions</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
            <p>Generated by AI Trading Bot Neural Network Analysis Tool</p>
            <p>For questions or improvements, refer to the documentation.</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)


def check_dependencies():
    """Check if required dependencies are available"""
    
    required_packages = [
        'torch', 'stable_baselines3', 'plotly', 'pandas', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


if __name__ == "__main__":
    print("Starting Neural Network Visualization Demo...")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies first.")
        sys.exit(1)
    
    # Run the demonstration
    run_visualization_demo()
    
    print("\n🎉 Demo completed! Check the 'neural_network_analysis' directory for results.")