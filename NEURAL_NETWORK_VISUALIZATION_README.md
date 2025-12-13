# Neural Network Visualization Tool

A comprehensive visualization tool for analyzing the RecurrentPPO model with LSTM layers used in the AI trading bot project.

## Overview

This tool provides multiple visualization types specifically designed for financial time series analysis and trading decisions:

1. **Model Architecture Visualization** - Complete network structure including LSTM layers and MLP heads
2. **Layer Weights Analysis** - Weight matrices and distributions for each layer
3. **LSTM Hidden States Tracking** - How hidden states evolve during trading sequences
4. **Feature Importance Evolution** - How feature importance changes during training
5. **Decision Flow Visualization** - Trace inputs flowing through the network to final decisions
6. **Network Activation Patterns** - Visualize neuron activations for different market conditions

## Features

### 🎯 Core Capabilities
- **Modular Design**: Each visualization can be used independently
- **Interactive Visualizations**: Uses Plotly for interactive charts
- **Integration Ready**: Works with existing RecurrentPPO model structure
- **Financial Focus**: Specifically designed for trading bot analysis
- **Comprehensive Reports**: Automatic generation of complete analysis reports

### 📊 Visualization Types

#### 1. Model Architecture
- Complete network structure diagram
- Layer connectivity visualization
- Component identification (LSTM, MLP, Actor, Critic)
- Interactive network topology

#### 2. Layer Weights Analysis
- Weight distribution histograms
- Statistical analysis (mean, std, variance)
- Layer-by-layer weight comparison
- Potential overfitting detection

#### 3. LSTM Hidden States
- Hidden state evolution over time
- Cell state tracking
- Neuron activation patterns
- Temporal dynamics analysis

#### 4. Feature Importance Evolution
- Training progress tracking
- Feature importance ranking
- Temporal importance changes
- Top feature identification

#### 5. Decision Flow Analysis
- Input-to-output flow visualization
- Feature contribution analysis
- Decision path tracing
- Captum integration for attribution

#### 6. Activation Patterns
- Neuron activation heatmaps
- Market condition-based analysis
- Layer activation distributions
- Temporal activation patterns

## Installation

### Requirements
```bash
pip install torch stable-baselines3 plotly pandas numpy matplotlib seaborn
```

### Optional Dependencies
```bash
pip install captum  # For advanced attribution analysis
```

### Project Dependencies
The tool integrates with existing trading bot components:
- `features.py` - Feature extraction logic
- `enhanced_trading_env.py` - Environment structure
- `callbacks/recurrent_saliency.py` - Existing saliency analysis

## Usage

### Basic Usage

```python
from neural_network_visualizer import NeuralNetworkVisualizer

# Initialize with your trained model
visualizer = NeuralNetworkVisualizer(
    model=your_trained_model,
    feature_names=your_feature_names,
    save_dir="analysis_output"
)

# Create individual visualizations
arch_fig = visualizer.visualize_model_architecture()
weights_fig = visualizer.analyze_layer_weights()
lstm_fig = visualizer.track_lstm_hidden_states(observations)
```

### Complete Analysis Report

```python
# Generate comprehensive report
report_files = visualizer.create_comprehensive_report(
    observations=sample_observations,
    save_dir="neural_network_analysis"
)

# Access individual files
print("Generated files:")
for name, path in report_files.items():
    print(f"{name}: {path}")
```

### With Real Trading Data

```python
import pandas as pd
from features import get_features
from enhanced_trading_env import EnhancedTradingEnv

# Load your trained model
model = PPO.load("models/recurrent_ppo_trading_model.zip")

# Load trading data
df = pd.read_csv('BTCUSDT_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Extract features for visualization
observations = []
for timestamp in df.index[-50:]:  # Last 50 observations
    features = get_features(df, vp7_df, vp30_df, timestamp)
    observations.append(features)

observations = np.array(observations)

# Create visualizer
visualizer = NeuralNetworkVisualizer(model=model)
report_files = visualizer.create_comprehensive_report(observations=observations)
```

## Demo Script

Run the complete demonstration:

```bash
python neural_network_demo.py
```

This will:
1. Load or create a demo model
2. Generate sample observations using real feature extraction
3. Create all visualizations
4. Generate a comprehensive HTML report
5. Create an index file for easy navigation

## Integration with Existing Codebase

### With Callbacks

The visualizer integrates with existing callbacks for real-time analysis:

```python
from neural_network_visualizer import NeuralNetworkVisualizer
from callbacks.recurrent_saliency import RecurrentFeatureSaliencyCallback

# In your training script
visualizer = NeuralNetworkVisualizer(model=model)

# Add callback for periodic visualization
callback = RecurrentFeatureSaliencyCallback(
    check_freq=5000,
    save_path="./logs/visualizations"
)

# During training
model.learn(total_timesteps=100000, callback=callback)
```

### With Feature Importance Analysis

```python
# Combine with existing saliency analysis
from callbacks.feature_saliency import FeatureSaliencyCallback

# Create combined callback
class VisualizationCallback(BaseCallback):
    def __init__(self, visualizer, verbose=0):
        super().__init__(verbose)
        self.visualizer = visualizer
        
    def _on_step(self):
        if self.n_calls % 10000 == 0:
            # Generate visualizations periodically
            observations = self.locals['new_obs']
            self.visualizer.track_lstm_hidden_states(observations)
        return True
```

## Configuration

### Feature Names
Provide meaningful feature names for better visualization:

```python
feature_names = [
    "Price_Close", "Price_High", "Price_Low", "Volume",
    "RSI", "MACD", "EMA_20", "Bollinger_Upper",
    # ... 220 total features
]

visualizer = NeuralNetworkVisualizer(
    model=model,
    feature_names=feature_names
)
```

### Output Directory Structure
```
neural_network_analysis/
├── 00_summary_report.html       # Main report
├── 01_model_architecture.html   # Network structure
├── 02_layer_weights.html        # Weight analysis
├── 03_lstm_hidden_states.html   # LSTM dynamics
├── 04_activation_patterns.html  # Neural activations
├── 05_feature_importance_evolution.html
└── index.html                   # Navigation page
```

## Advanced Usage

### Custom Analysis
```python
# Analyze specific layers
lstm_weights = visualizer.extract_layer_weights('lstm_actor')
mlp_weights = visualizer.extract_layer_weights('mlp_extractor')

# Custom activation tracking
custom_activations = visualizer.track_custom_activations(
    observations, 
    target_layers=['lstm_actor', 'action_net']
)
```

### Batch Analysis
```python
# Analyze multiple models
models = {
    'baseline': baseline_model,
    'optimized': optimized_model,
    'latest': latest_model
}

for name, model in models.items():
    visualizer = NeuralNetworkVisualizer(model=model)
    report_files = visualizer.create_comprehensive_report(
        save_dir=f"comparison/{name}"
    )
```

## Interpreting Results

### Model Architecture
- **LSTM Layers**: Show temporal memory capabilities
- **MLP Heads**: Display processing complexity
- **Connections**: Reveal information flow

### Layer Weights
- **Weight Distributions**: Normal distributions indicate healthy training
- **Weight Magnitudes**: Large weights may indicate overfitting
- **Gradient Flow**: Consistent weights suggest stable learning

### LSTM Hidden States
- **Temporal Patterns**: Look for consistent activation patterns
- **State Evolution**: Smooth transitions indicate good temporal learning
- **Memory Retention**: Long-term dependencies in hidden states

### Feature Importance
- **Top Features**: Most influential inputs for decisions
- **Evolution**: How importance changes during training
- **Stability**: Consistent importance across time steps

### Decision Flow
- **Path Analysis**: How information flows from input to output
- **Bottlenecks**: Layers that heavily influence decisions
- **Attribution**: Which features contribute most to specific decisions

### Activation Patterns
- **Neuron Specialization**: Different neurons for different market conditions
- **Activation Distribution**: Healthy activation patterns
- **Temporal Correlation**: Consistent activation across similar inputs

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```python
   # Ensure model is properly saved
   model.save("models/my_model")
   
   # Load with correct policy
   model = PPO.load("models/my_model")
   ```

2. **Feature Count Mismatch**
   ```python
   # Check feature dimensions
   print(f"Model expects: {model.observation_space.shape}")
   print(f"Features provided: {len(feature_names)}")
   ```

3. **CUDA Memory Issues**
   ```python
   # Move model to CPU for visualization
   model = model.cpu()
   ```

4. **Missing Dependencies**
   ```bash
   # Install all required packages
   pip install -r requirements.txt
   ```

### Performance Tips

1. **Reduce Observation Count**: Limit analysis to representative samples
2. **Batch Processing**: Process multiple observations together
3. **Selective Analysis**: Focus on most interesting layers/features
4. **Memory Management**: Clear GPU memory between analyses

## API Reference

### NeuralNetworkVisualizer

#### Methods
- `visualize_model_architecture(save_path=None)` - Network structure
- `analyze_layer_weights(layer_name=None, save_path=None)` - Weight analysis
- `track_lstm_hidden_states(observations, episode_starts=None, save_path=None)` - LSTM tracking
- `visualize_feature_importance_evolution(training_steps, importance_scores, save_path=None)` - Feature importance
- `visualize_decision_flow(observation, feature_importance=None, save_path=None)` - Decision analysis
- `visualize_activation_patterns(observations, market_conditions=None, save_path=None)` - Activation patterns
- `create_comprehensive_report(observations=None, save_dir=None)` - Complete analysis

#### Parameters
- `model`: Trained RecurrentPPO model
- `feature_names`: List of 220 feature names
- `save_dir`: Output directory for visualizations

## Contributing

To extend the visualization tool:

1. **Add New Visualizations**: Inherit from base class
2. **Custom Analysis**: Add specialized analysis methods
3. **Integration**: Connect with existing callbacks
4. **Testing**: Add unit tests for new features

## License

This tool is part of the AI Trading Bot project and follows the same licensing terms.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the demo script examples
3. Examine the existing callback implementations
4. Refer to the Stable-Baselines3 documentation