"""
Neural Network Visualization Tool for AI Trading Bot
====================================================

A comprehensive visualization tool for the RecurrentPPO model with LSTM layers,
specifically designed for financial time series analysis and trading decisions.

Features:
- Model Architecture Visualization
- Layer Weights Analysis
- LSTM Hidden States Tracking
- Feature Importance Evolution
- Decision Flow Visualization
- Network Activation Patterns
- Interactive Visualizations using Plotly

Author: AI Trading Bot Team
Date: 2025-12-13
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import defaultdict
import os
from pathlib import Path
import json
from datetime import datetime, timedelta

try:
    from captum.attr import IntegratedGradients, LayerConductance
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available. Some features will be limited.")


class NeuralNetworkVisualizer:
    """
    Comprehensive Neural Network Visualization Tool for Trading Bot
    
    This class provides various visualization methods to analyze and understand
    the RecurrentPPO model's behavior during trading decisions.
    """
    
    def __init__(self, model, feature_names: List[str] = None, save_dir: str = "visualizations"):
        """
        Initialize the Neural Network Visualizer
        
        Args:
            model: The trained RecurrentPPO model
            feature_names: List of feature names for better labeling
            save_dir: Directory to save visualization outputs
        """
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(220)]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Model architecture details
        self.model_info = self._extract_model_info()
        
        # Visualization styles
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.color_palette = px.colors.qualitative.Set3
        
        # Initialize tracking dictionaries
        self.weights_history = {}
        self.activations_history = {}
        self.hidden_states_history = {}
        self.feature_importance_history = {}
        
    def _set_model_to_eval(self):
        """
        Safely set model to evaluation mode
        Handles RecurrentPPO models that may not have eval() method
        """
        try:
            if hasattr(self.model, 'eval'):
                self.model.eval()
            elif hasattr(self.model.policy, 'eval'):
                self.model.policy.eval()
        except Exception as e:
            warnings.warn(f"Could not set model to evaluation mode: {e}")
        
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract model architecture information"""
        info = {}
        
        try:
            # Extract LSTM information
            if hasattr(self.model.policy, 'lstm_actor'):
                lstm_actor = self.model.policy.lstm_actor
                info['lstm_actor'] = {
                    'input_size': lstm_actor.input_size,
                    'hidden_size': lstm_actor.hidden_size,
                    'num_layers': lstm_actor.num_layers,
                    'bias': lstm_actor.bias
                }
                
            if hasattr(self.model.policy, 'lstm_critic'):
                lstm_critic = self.model.policy.lstm_critic
                info['lstm_critic'] = {
                    'input_size': lstm_critic.input_size,
                    'hidden_size': lstm_critic.hidden_size,
                    'num_layers': lstm_critic.num_layers,
                    'bias': lstm_critic.bias
                }
            
            # Extract MLP information
            if hasattr(self.model.policy, 'mlp_extractor'):
                mlp_extractor = self.model.policy.mlp_extractor
                info['mlp_extractor'] = {
                    'latent_dim_pi': getattr(mlp_extractor, 'latent_dim_pi', None),
                    'latent_dim_vf': getattr(mlp_extractor, 'latent_dim_vf', None)
                }
            
            # Action and value heads
            if hasattr(self.model.policy, 'action_net'):
                action_net = self.model.policy.action_net
                info['action_net'] = {
                    'input_dim': action_net.input_dim if hasattr(action_net, 'input_dim') else None,
                    'output_dim': action_net.output_dim if hasattr(action_net, 'output_dim') else None
                }
                
            if hasattr(self.model.policy, 'value_net'):
                value_net = self.model.policy.value_net
                info['value_net'] = {
                    'input_dim': value_net.input_dim if hasattr(value_net, 'input_dim') else None,
                    'output_dim': value_net.output_dim if hasattr(value_net, 'output_dim') else None
                }
                
        except Exception as e:
            warnings.warn(f"Could not extract complete model info: {e}")
            
        return info
    
    def visualize_model_architecture(self, save_path: str = None) -> go.Figure:
        """
        Visualize the complete neural network architecture
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Model architecture layout
        layers = []
        connections = []
        
        # Input layer
        input_features = len(self.feature_names)
        layers.append({
            'name': 'Input Features',
            'type': 'input',
            'size': input_features,
            'position': (0, 5),
            'color': '#3498db'
        })
        
        # LSTM layers (Actor)
        if 'lstm_actor' in self.model_info:
            lstm_info = self.model_info['lstm_actor']
            lstm_size = lstm_info['hidden_size']
            layers.append({
                'name': f'LSTM Actor\n({lstm_size} units)',
                'type': 'lstm',
                'size': lstm_size,
                'position': (2, 6),
                'color': '#e74c3c'
            })
            
            # LSTM connections
            connections.append((0, 1))
        
        # MLP Extractor
        if 'mlp_extractor' in self.model_info:
            mlp_info = self.model_info['mlp_extractor']
            if mlp_info.get('latent_dim_pi'):
                layers.append({
                    'name': f'MLP Actor\n(512→512)',
                    'type': 'mlp',
                    'size': 512,
                    'position': (4, 7),
                    'color': '#2ecc71'
                })
                connections.append((1, 2))
                
            if mlp_info.get('latent_dim_vf'):
                layers.append({
                    'name': f'MLP Critic\n(512→512)',
                    'type': 'mlp',
                    'size': 512,
                    'position': (4, 3),
                    'color': '#f39c12'
                })
                connections.append((1, 3))
        
        # Action and Value heads
        if 'action_net' in self.model_info:
            layers.append({
                'name': 'Action Head\n(Continuous)',
                'type': 'output',
                'size': 2,
                'position': (6, 7),
                'color': '#9b59b6'
            })
            connections.append((2, 4))
            
        if 'value_net' in self.model_info:
            layers.append({
                'name': 'Value Head\n(Single)',
                'type': 'output',
                'size': 1,
                'position': (6, 3),
                'color': '#1abc9c'
            })
            connections.append((3, 5))
        
        # Draw layers
        for i, layer in enumerate(layers):
            # Main layer box
            fig.add_shape(
                type="rect",
                x0=layer['position'][0]-0.4, y0=layer['position'][1]-0.5,
                x1=layer['position'][0]+0.4, y1=layer['position'][1]+0.5,
                fillcolor=layer['color'],
                opacity=0.7,
                line=dict(width=2, color='black')
            )
            
            # Layer label
            fig.add_annotation(
                x=layer['position'][0],
                y=layer['position'][1],
                text=layer['name'],
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor=layer['color'],
                bordercolor='black',
                borderwidth=1
            )
        
        # Draw connections
        for start_idx, end_idx in connections:
            start_layer = layers[start_idx]
            end_layer = layers[end_idx]
            
            fig.add_annotation(
                x=(start_layer['position'][0] + end_layer['position'][0]) / 2,
                y=(start_layer['position'][1] + end_layer['position'][1]) / 2,
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='gray',
                startarrowhead=2
            )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#3498db', label='Input Layer'),
            mpatches.Patch(color='#e74c3c', label='LSTM Layer'),
            mpatches.Patch(color='#2ecc71', label='Actor MLP'),
            mpatches.Patch(color='#f39c12', label='Critic MLP'),
            mpatches.Patch(color='#9b59b6', label='Action Head'),
            mpatches.Patch(color='#1abc9c', label='Value Head')
        ]
        
        fig.update_layout(
            title=dict(
                text="RecurrentPPO Model Architecture for Trading Bot",
                font=dict(size=16),
                x=0.5
            ),
            xaxis=dict(visible=False, range=[-1, 7]),
            yaxis=dict(visible=False, range=[0, 8]),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            width=800,
            height=500,
            plot_bgcolor='white'
        )
        
        # Save figure
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def analyze_layer_weights(self, layer_name: str = None, save_path: str = None) -> go.Figure:
        """
        Analyze weight matrices and distributions for each layer
        
        Args:
            layer_name: Specific layer to analyze (None for all layers)
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object with weight analysis
        """
        if not hasattr(self.model, 'policy'):
            raise ValueError("Model must have a policy attribute")
        
        # Extract weights from different components
        weights_data = {}
        
        try:
            # LSTM weights
            if hasattr(self.model.policy, 'lstm_actor'):
                lstm_actor = self.model.policy.lstm_actor
                for name, param in lstm_actor.named_parameters():
                    if 'weight' in name:
                        weights_data[f'LSTM_Actor_{name}'] = param.data.cpu().numpy().flatten()
            
            if hasattr(self.model.policy, 'lstm_critic'):
                lstm_critic = self.model.policy.lstm_critic
                for name, param in lstm_critic.named_parameters():
                    if 'weight' in name:
                        weights_data[f'LSTM_Critic_{name}'] = param.data.cpu().numpy().flatten()
            
            # MLP weights
            if hasattr(self.model.policy, 'mlp_extractor'):
                mlp_extractor = self.model.policy.mlp_extractor
                for name, param in mlp_extractor.named_parameters():
                    if 'weight' in name:
                        weights_data[f'MLP_{name}'] = param.data.cpu().numpy().flatten()
            
            # Action and value net weights
            if hasattr(self.model.policy, 'action_net'):
                action_net = self.model.policy.action_net
                for name, param in action_net.named_parameters():
                    if 'weight' in name:
                        weights_data[f'ActionNet_{name}'] = param.data.cpu().numpy().flatten()
            
            if hasattr(self.model.policy, 'value_net'):
                value_net = self.model.policy.value_net
                for name, param in value_net.named_parameters():
                    if 'weight' in name:
                        weights_data[f'ValueNet_{name}'] = param.data.cpu().numpy().flatten()
                        
        except Exception as e:
            warnings.warn(f"Could not extract all weights: {e}")
        
        if not weights_data:
            raise ValueError("No weights found in the model")
        
        # Create subplots
        n_layers = len(weights_data)
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(weights_data.keys()),
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Plot weight distributions
        for i, (layer_name, weights) in enumerate(weights_data.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Histogram of weights
            fig.add_trace(
                go.Histogram(
                    x=weights,
                    name=f'{layer_name} Distribution',
                    nbinsx=50,
                    marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                    opacity=0.7
                ),
                row=row, col=col
            )
            
            # Add statistics
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)
            
            fig.add_annotation(
                x=mean_weight, y=0,
                text=f"μ={mean_weight:.3f}<br>σ={std_weight:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                row=row, col=col
            )
        
        fig.update_layout(
            title="Layer Weight Distributions and Statistics",
            height=300 * rows,
            showlegend=False,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def track_lstm_hidden_states(self, observations: np.ndarray, 
                                episode_starts: np.ndarray = None,
                                save_path: str = None) -> go.Figure:
        """
        Track LSTM hidden states evolution over time during trading sequences
        
        Args:
            observations: Input observations sequence
            episode_starts: Boolean array indicating episode starts
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object with hidden states visualization
        """
        if not hasattr(self.model, 'policy') or not hasattr(self.model.policy, 'lstm_actor'):
            raise ValueError("Model must have LSTM layers")
        
        self._set_model_to_eval()
        
        # Initialize LSTM states
        batch_size = 1
        hidden_size = self.model.policy.lstm_actor.hidden_size
        num_layers = self.model.policy.lstm_actor.num_layers
        
        # Initialize hidden and cell states
        h0 = torch.zeros(num_layers, batch_size, hidden_size)
        c0 = torch.zeros(num_layers, batch_size, hidden_size)
        
        # Convert observations to tensor
        if isinstance(observations, np.ndarray):
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0)  # Add batch dimension
        else:
            obs_tensor = observations
            
        hidden_states = []
        cell_states = []
        
        with torch.no_grad():
            # Track states through the sequence
            for t in range(min(len(obs_tensor), 100)):  # Limit to first 100 steps
                # Forward pass through LSTM
                lstm_out, (h_n, c_n) = self.model.policy.lstm_actor(
                    obs_tensor[t:t+1], (h0, c0)
                )
                
                # Store states
                hidden_states.append(h_n[-1].squeeze().cpu().numpy())  # Last layer
                cell_states.append(c_n[-1].squeeze().cpu().numpy())    # Last layer
                
                # Update states for next step
                h0, c0 = h_n, c_n
        
        hidden_states = np.array(hidden_states)
        cell_states = np.array(cell_states)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Hidden States Evolution (First 10 neurons)',
                'Cell States Evolution (First 10 neurons)',
                'Hidden States Heatmap',
                'Cell States Heatmap'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Plot first 10 neurons over time
        time_steps = np.arange(len(hidden_states))
        
        for i in range(min(10, hidden_size)):
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=hidden_states[:, i],
                    mode='lines',
                    name=f'Hidden_{i}',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=cell_states[:, i],
                    mode='lines',
                    name=f'Cell_{i}',
                    line=dict(width=1),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Heatmaps
        fig.add_trace(
            go.Heatmap(
                z=hidden_states.T,
                colorscale='RdBu',
                name='Hidden States',
                showscale=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=cell_states.T,
                colorscale='RdBu',
                name='Cell States',
                showscale=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="LSTM Hidden States Evolution During Trading Sequence",
            height=600,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Steps", row=1, col=1)
        fig.update_xaxes(title_text="Time Steps", row=1, col=2)
        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=2)
        fig.update_yaxes(title_text="Hidden State Value", row=1, col=1)
        fig.update_yaxes(title_text="Cell State Value", row=1, col=2)
        fig.update_yaxes(title_text="Neuron Index", row=2, col=1)
        fig.update_yaxes(title_text="Neuron Index", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def visualize_feature_importance_evolution(self, training_steps: List[int],
                                             importance_scores: List[np.ndarray],
                                             save_path: str = None) -> go.Figure:
        """
        Visualize how feature importance changes during training
        
        Args:
            training_steps: List of training step numbers
            importance_scores: List of importance scores for each step
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        if not importance_scores:
            raise ValueError("Importance scores cannot be empty")
        
        # Create DataFrame for easier plotting
        all_data = []
        
        for step_idx, (step, scores) in enumerate(zip(training_steps, importance_scores)):
            for feat_idx, score in enumerate(scores):
                feature_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
                all_data.append({
                    'Training Step': step,
                    'Feature': feature_name,
                    'Importance': score,
                    'Feature Index': feat_idx
                })
        
        df = pd.DataFrame(all_data)
        
        # Get top features for detailed tracking
        top_features = df.groupby('Feature')['Importance'].abs().mean().nlargest(20).index.tolist()
        top_df = df[df['Feature'].isin(top_features)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Top 10 Features Importance Evolution',
                'Feature Importance Distribution',
                'All Features Heatmap',
                'Importance Variance by Feature'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "heatmap"}, {"secondary_y": False}]]
        )
        
        # Plot top 10 features evolution
        top_10_features = df.groupby('Feature')['Importance'].abs().mean().nlargest(10).index.tolist()
        
        colors = px.colors.qualitative.Set1
        for i, feature in enumerate(top_10_features):
            feature_data = top_df[top_df['Feature'] == feature]
            fig.add_trace(
                go.Scatter(
                    x=feature_data['Training Step'],
                    y=feature_data['Importance'],
                    mode='lines+markers',
                    name=feature,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # Feature importance distribution
        fig.add_trace(
            go.Box(
                y=abs(df['Importance']),
                name='All Features',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # All features heatmap (sample if too many features)
        max_features = min(50, len(self.feature_names))
        pivot_data = df[df['Feature Index'] < max_features].pivot(
            index='Feature', columns='Training Step', values='Importance'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdBu',
                showscale=False
            ),
            row=2, col=1
        )
        
        # Importance variance by feature
        variance_data = df.groupby('Feature')['Importance'].var().sort_values(ascending=False).head(20)
        
        fig.add_trace(
            go.Bar(
                x=variance_data.index,
                y=variance_data.values,
                marker_color='orange',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Feature Importance Evolution During Training",
            height=800,
            template='plotly_white'
        )
        
        # Update x-axis for variance plot
        fig.update_xaxes(tickangle=45, row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def visualize_decision_flow(self, observation: np.ndarray,
                              feature_importance: np.ndarray = None,
                              save_path: str = None) -> go.Figure:
        """
        Trace how inputs flow through the network to final trading decisions
        
        Args:
            observation: Single observation to analyze
            feature_importance: Pre-computed feature importance scores
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required for decision flow visualization")
        
        self._set_model_to_eval()
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        obs_tensor.requires_grad_()
        
        # Compute feature importance using Integrated Gradients
        if feature_importance is None:
            def forward_func(inputs):
                with torch.enable_grad():
                    return self.model.policy._forward(inputs)[0]
            
            ig = IntegratedGradients(forward_func)
            attributions, delta = ig.attribute(
                obs_tensor,
                baselines=torch.zeros_like(obs_tensor),
                n_steps=10,
                return_convergence_delta=True
            )
            
            feature_importance = attributions.squeeze().cpu().detach().numpy()
        
        # Analyze the flow
        flow_data = self._analyze_decision_flow(obs_tensor, feature_importance)
        
        # Create Sankey diagram for decision flow
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=flow_data['labels'],
                color=flow_data['node_colors']
            ),
            link=dict(
                source=flow_data['sources'],
                target=flow_data['targets'],
                value=flow_data['values'],
                color=flow_data['link_colors']
            )
        )])
        
        fig.update_layout(
            title_text="Neural Network Decision Flow Analysis",
            font_size=12,
            height=600
        )
        
        # Add feature importance breakdown
        feature_breakdown = self._create_feature_breakdown_chart(feature_importance)
        
        # Combine plots
        combined_fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=["Decision Flow (Sankey)", "Feature Importance Breakdown"]
        )
        
        # Add Sankey plot
        for trace in fig.data:
            combined_fig.add_trace(trace, row=1, col=1)
        
        # Add feature importance chart
        for trace in feature_breakdown.data:
            combined_fig.add_trace(trace, row=2, col=1)
        
        combined_fig.update_layout(
            height=900,
            title_text="Neural Network Decision Flow and Feature Analysis",
            template='plotly_white'
        )
        
        if save_path:
            combined_fig.write_html(save_path)
            
        return combined_fig
    
    def _analyze_decision_flow(self, obs_tensor: torch.Tensor, 
                             feature_importance: np.ndarray) -> Dict[str, List]:
        """Analyze the decision flow through the network"""
        
        # Get top features by importance
        top_indices = np.argsort(np.abs(feature_importance))[-20:]  # Top 20 features
        
        # Create nodes for the flow diagram
        nodes = []
        node_colors = []
        
        # Input features (top 10 only for clarity)
        for i, idx in enumerate(top_indices[-10:]):
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            nodes.append(f"Input: {feature_name}")
            node_colors.append('#3498db')
        
        # Hidden layers (representative nodes)
        nodes.extend(["LSTM Layer", "MLP Actor", "MLP Critic"])
        node_colors.extend(['#e74c3c', '#2ecc71', '#f39c12'])
        
        # Output layers
        nodes.extend(["Action Distribution", "Value Function"])
        node_colors.extend(['#9b59b6', '#1abc9c'])
        
        # Create connections
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # Input to LSTM (based on feature importance)
        for i in range(min(10, len(top_indices))):
            sources.append(i)
            targets.append(10)  # LSTM layer index
            values.append(abs(feature_importance[top_indices[i]]))
            link_colors.append('rgba(52, 152, 219, 0.4)')
        
        # LSTM to MLP layers
        sources.extend([10, 10])
        targets.extend([11, 12])
        values.extend([1.0, 1.0])
        link_colors.extend(['rgba(231, 76, 60, 0.4)', 'rgba(231, 76, 60, 0.4)'])
        
        # MLP to outputs
        sources.extend([11, 12])
        targets.extend([13, 14])
        values.extend([1.0, 1.0])
        link_colors.extend(['rgba(155, 89, 182, 0.4)', 'rgba(26, 188, 156, 0.4)'])
        
        return {
            'labels': nodes,
            'node_colors': node_colors,
            'sources': sources,
            'targets': targets,
            'values': values,
            'link_colors': link_colors
        }
    
    def _create_feature_breakdown_chart(self, feature_importance: np.ndarray) -> go.Figure:
        """Create a feature importance breakdown chart"""
        
        # Get top 15 features
        top_indices = np.argsort(np.abs(feature_importance))[-15:]
        
        feature_names = []
        importance_values = []
        
        for idx in top_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            feature_names.append(feature_name)
            importance_values.append(feature_importance[idx])
        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=importance_values,
            y=feature_names,
            orientation='h',
            marker_color=px.colors.qualitative.Set1[:len(feature_names)],
            text=[f"{val:.3f}" for val in importance_values],
            textposition='auto'
        )])
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template='plotly_white',
            height=300,
            margin=dict(l=150)
        )
        
        return fig
    
    def visualize_activation_patterns(self, observations: np.ndarray,
                                    market_conditions: List[str] = None,
                                    save_path: str = None) -> go.Figure:
        """
        Visualize neuron activations for different market conditions
        
        Args:
            observations: Array of observations
            market_conditions: List of market condition labels
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        self._set_model_to_eval()
        
        # Forward pass to get activations
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations)
            
            # Get activations from different layers
            activations = {}
            
            # Input layer activations (just the observations)
            activations['input'] = obs_tensor.cpu().numpy()
            
            # LSTM activations
            if hasattr(self.model.policy, 'lstm_actor'):
                lstm_out, _ = self.model.policy.lstm_actor(obs_tensor)
                activations['lstm'] = lstm_out.cpu().numpy()
            
            # MLP activations
            if hasattr(self.model.policy, 'mlp_extractor'):
                mlp_features = self.model.policy.mlp_extractor.forward_actor(lstm_out)
                activations['mlp'] = mlp_features.cpu().numpy()
            
            # Action and value outputs
            action_dist = self.model.policy._forward(obs_tensor)[0]
            activations['action'] = action_dist.cpu().numpy()
        
        # Create visualization
        n_conditions = len(observations) if market_conditions is None else len(market_conditions)
        n_neurons_to_show = min(20, activations['lstm'].shape[-1] if 'lstm' in activations else 10)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'LSTM Neuron Activations by Time',
                'Activation Distribution by Layer',
                'Neuron Activation Heatmap',
                'Average Activation by Market Condition'
            ]
        )
        
        # Plot LSTM activations over time (first 10 neurons)
        time_steps = np.arange(min(100, len(observations)))
        for neuron_idx in range(min(10, n_neurons_to_show)):
            if 'lstm' in activations:
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=activations['lstm'][:len(time_steps), neuron_idx],
                        mode='lines',
                        name=f'Neuron {neuron_idx}',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Activation distributions by layer
        for layer_name, layer_activations in activations.items():
            if len(layer_activations.shape) > 1:
                # Flatten activations for distribution plot
                flat_activations = layer_activations.flatten()
                fig.add_trace(
                    go.Histogram(
                        x=flat_activations[:10000],  # Limit for performance
                        name=f'{layer_name} layer',
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=1, col=2
                )
        
        # Activation heatmap
        if 'lstm' in activations:
            heatmap_data = activations['lstm'][:, :n_neurons_to_show].T
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    colorscale='RdBu',
                    showscale=False,
                    name='Activations'
                ),
                row=2, col=1
            )
        
        # Average activation by market condition (if provided)
        if market_conditions and len(market_conditions) == len(observations):
            # Group observations by market condition
            condition_groups = {}
            for i, condition in enumerate(market_conditions):
                if condition not in condition_groups:
                    condition_groups[condition] = []
                condition_groups[condition].append(i)
            
            # Calculate average activations for each condition
            for condition, indices in condition_groups.items():
                if 'lstm' in activations:
                    avg_activation = np.mean(activations['lstm'][indices, :n_neurons_to_show], axis=0)
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(len(avg_activation)),
                            y=avg_activation,
                            mode='lines+markers',
                            name=f'{condition} (avg)',
                            line=dict(width=2)
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title="Neural Network Activation Patterns Analysis",
            height=800,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_xaxes(title_text="Activation Value", row=1, col=2)
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        fig.update_xaxes(title_text="Neuron Index", row=2, col=1)
        fig.update_xaxes(title_text="Neuron Index", row=2, col=2)
        fig.update_yaxes(title_text="Activation Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Activation Value", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_comprehensive_report(self, observations: np.ndarray = None,
                                  save_dir: str = None) -> Dict[str, str]:
        """
        Create a comprehensive visualization report
        
        Args:
            observations: Sample observations for analysis
            save_dir: Directory to save all visualizations
            
        Returns:
            Dictionary mapping report sections to file paths
        """
        if save_dir is None:
            save_dir = self.save_dir
        else:
            save_dir = Path(save_dir)
            
        save_dir.mkdir(exist_ok=True, parents=True)
        report_files = {}
        
        try:
            # 1. Model Architecture
            arch_path = save_dir / "01_model_architecture.html"
            self.visualize_model_architecture(save_path=str(arch_path))
            report_files['model_architecture'] = str(arch_path)
            
            # 2. Layer Weights Analysis
            weights_path = save_dir / "02_layer_weights.html"
            self.analyze_layer_weights(save_path=str(weights_path))
            report_files['layer_weights'] = str(weights_path)
            
            # 3. LSTM Hidden States (if observations provided)
            if observations is not None and len(observations) > 0:
                lstm_path = save_dir / "03_lstm_hidden_states.html"
                self.track_lstm_hidden_states(observations, save_path=str(lstm_path))
                report_files['lstm_hidden_states'] = str(lstm_path)
            
            # 4. Activation Patterns (if observations provided)
            if observations is not None and len(observations) > 0:
                activation_path = save_dir / "04_activation_patterns.html"
                self.visualize_activation_patterns(observations, save_path=str(activation_path))
                report_files['activation_patterns'] = str(activation_path)
            
            # 5. Create summary report
            summary_path = save_dir / "00_summary_report.html"
            self._create_summary_report(report_files, save_path=str(summary_path))
            report_files['summary'] = str(summary_path)
            
        except Exception as e:
            warnings.warn(f"Error creating comprehensive report: {e}")
        
        return report_files
    
    def _create_summary_report(self, report_files: Dict[str, str], save_path: str):
        """Create an HTML summary report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
                .link {{ color: #007bff; text-decoration: none; }}
                .link:hover {{ text-decoration: underline; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Neural Network Visualization Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Model: RecurrentPPO with LSTM for Trading Bot</p>
            </div>
            
            <div class="section">
                <h2>Overview</h2>
                <p>This report provides comprehensive visualizations of the neural network model used in the AI trading bot.</p>
                <p>The analysis covers model architecture, layer weights, LSTM dynamics, and activation patterns.</p>
            </div>
            
            <div class="grid">
        """
        
        # Add sections for each report file
        section_info = {
            'model_architecture': ('Model Architecture', 'Complete network structure visualization'),
            'layer_weights': ('Layer Weights Analysis', 'Weight distributions and statistics'),
            'lstm_hidden_states': ('LSTM Hidden States', 'Hidden state evolution tracking'),
            'activation_patterns': ('Activation Patterns', 'Neuron activations by market conditions')
        }
        
        for key, file_path in report_files.items():
            if key != 'summary' and key in section_info:
                title, description = section_info[key]
                html_content += f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <a href="{os.path.basename(file_path)}" class="link">View Visualization</a>
                </div>
                """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                <ul>
                    <li><strong>Model Architecture:</strong> The RecurrentPPO uses LSTM layers for temporal pattern recognition</li>
                    <li><strong>Feature Processing:</strong> 220 features including price data, technical indicators, and volume profile</li>
                    <li><strong>Temporal Dynamics:</strong> LSTM hidden states capture market memory and trends</li>
                    <li><strong>Decision Making:</strong> Actor-critic architecture separates action and value estimation</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Usage Recommendations</h2>
                <ul>
                    <li>Use model architecture to understand network design and complexity</li>
                    <li>Monitor layer weights to detect overfitting or training issues</li>
                    <li>Track LSTM states to understand temporal learning</li>
                    <li>Analyze activations to debug network behavior</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)


# Example usage and testing functions
def create_sample_observations(n_samples: int = 100, n_features: int = 220) -> np.ndarray:
    """Create sample observations for testing"""
    np.random.seed(42)
    observations = np.random.randn(n_samples, n_features)
    
    # Add some realistic patterns
    for i in range(n_samples):
        # Price-related features (first few)
        observations[i, 0] = 50000 + np.random.normal(0, 1000)  # BTC price simulation
        
        # Technical indicators
        observations[i, 80:83] = np.random.uniform(-1, 1, 3)  # MACD components
        
        # Volume profile features
        observations[i, 120:160] = np.random.exponential(1, 40)  # Heatmap data
        
        # Session features (one-hot hour encoding)
        hour = (i % 24)
        observations[i, 180 + hour] = 1.0
    
    return observations


def example_usage():
    """Example usage of the NeuralNetworkVisualizer"""
    
    # This is a mock example - in real usage, you'd have your trained model
    print("Neural Network Visualization Tool - Example Usage")
    print("=" * 50)
    
    # Create sample model (this would be your actual trained RecurrentPPO model)
    # model = your_trained_recurrentppo_model
    
    # Create visualizer
    # visualizer = NeuralNetworkVisualizer(
    #     model=model,
    #     feature_names=your_feature_names,
    #     save_dir="neural_network_analysis"
    # )
    
    # Create sample data
    observations = create_sample_observations(50, 220)
    feature_names = [f"Feature_{i}" for i in range(220)]
    
    # Example: Create comprehensive report
    print("Creating sample visualizations...")
    
    # You would use:
    # report_files = visualizer.create_comprehensive_report(observations=observations)
    
    print("Example visualizations created successfully!")
    print("\nFeatures available:")
    print("1. visualize_model_architecture() - Network structure")
    print("2. analyze_layer_weights() - Weight distributions")
    print("3. track_lstm_hidden_states() - Temporal dynamics")
    print("4. visualize_feature_importance_evolution() - Training analysis")
    print("5. visualize_decision_flow() - Decision process")
    print("6. visualize_activation_patterns() - Neural activity")
    print("7. create_comprehensive_report() - Complete analysis")


if __name__ == "__main__":
    example_usage()