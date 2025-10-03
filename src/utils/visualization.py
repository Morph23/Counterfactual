"""
Visualization Utilities
Functions for visualizing counterfactual explanations and model predictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class CounterfactualVisualizer:
    """Visualization utilities for counterfactual explanations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: DPI for matplotlib figures
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
    
    def plot_image_counterfactual(self, 
                                result: Dict,
                                class_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Visualize image counterfactual results.
        
        Args:
            result: Result dictionary from counterfactual generation
            class_names: Names of classes
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result.get('original_image', np.zeros((224, 224, 3))))
        axes[0].set_title(f'Original Image\nPredicted: {class_names[result["original_class"]]}\n'
                         f'Confidence: {result["original_prediction"][result["original_class"]]:.3f}')
        axes[0].axis('off')
        
        # Counterfactual image
        axes[1].imshow(result['counterfactual'])
        axes[1].set_title(f'Counterfactual\nPredicted: {class_names[result["counterfactual_class"]]}\n'
                         f'Confidence: {result["counterfactual_prediction"][result["counterfactual_class"]]:.3f}')
        axes[1].axis('off')
        
        # Perturbation (difference)
        perturbation = result['perturbation']
        perturbation_norm = np.abs(perturbation)
        perturbation_norm = (perturbation_norm - perturbation_norm.min()) / (perturbation_norm.max() - perturbation_norm.min() + 1e-8)
        
        im = axes[2].imshow(perturbation_norm, cmap='hot')
        axes[2].set_title(f'Perturbation Magnitude\nL2 Distance: {result["l2_distance"]:.4f}')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_multiple_counterfactuals(self,
                                    results: List[Dict],
                                    feature_names: Optional[List[str]] = None,
                                    method_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Compare multiple counterfactual methods.
        
        Args:
            results: List of result dictionaries
            feature_names: Names of features
            method_names: Names of methods
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if method_names is None:
            method_names = [f'Method {i+1}' for i in range(len(results))]
        
        # Extract metrics
        methods = []
        l2_distances = []
        l1_distances = []
        success_rates = []
        
        for i, result in enumerate(results):
            if isinstance(result, list):  # Multiple counterfactuals from same method
                methods.append(method_names[i])
                l2_distances.append(np.mean([r.get('l2_distance', 0) for r in result if r.get('success', False)]))
                l1_distances.append(np.mean([r.get('l1_distance', 0) for r in result if r.get('success', False)]))
                success_rates.append(np.mean([r.get('success', False) for r in result]))
            else:  # Single counterfactual
                methods.append(method_names[i])
                l2_distances.append(result.get('l2_distance', 0))
                l1_distances.append(result.get('l1_distance', 0))
                success_rates.append(1.0 if result.get('success', False) else 0.0)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # L2 distances
        axes[0, 0].bar(methods, l2_distances, alpha=0.7)
        axes[0, 0].set_title('L2 Distance (Proximity)')
        axes[0, 0].set_ylabel('L2 Distance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # L1 distances
        axes[0, 1].bar(methods, l1_distances, alpha=0.7)
        axes[0, 1].set_title('L1 Distance (Sparsity)')
        axes[0, 1].set_ylabel('L1 Distance')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rates
        axes[1, 0].bar(methods, success_rates, alpha=0.7)
        axes[1, 0].set_title('Success Rate')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined score (example: success_rate / (1 + l2_distance))
        combined_scores = [sr / (1 + l2) if l2 > 0 else sr for sr, l2 in zip(success_rates, l2_distances)]
        axes[1, 1].bar(methods, combined_scores, alpha=0.7)
        axes[1, 1].set_title('Combined Score (Success / (1 + L2))')
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_feature_importance_counterfactual(self,
                                             result: Dict,
                                             feature_names: Optional[List[str]] = None,
                                             top_k: int = 10,
                                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance based on counterfactual changes.
        
        Args:
            result: Result dictionary from counterfactual generation
            feature_names: Names of features
            top_k: Number of top features to show
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        changes = result.get('feature_changes', {})
        
        if 'changes' in changes:
            feature_changes = changes['changes']
        else:
            # Fallback: calculate from arrays
            original = result.get('original_instance', np.array([]))
            counterfactual = result['counterfactual']
            
            if isinstance(original, dict):
                feature_changes = {}
                for key in original.keys():
                    cf_val = result.get('counterfactual_dict', {}).get(key, 0)
                    feature_changes[key] = cf_val - original[key]
            else:
                if feature_names is None:
                    feature_names = [f'Feature_{i}' for i in range(len(original))]
                feature_changes = {
                    feature_names[i]: counterfactual[i] - original[i] 
                    for i in range(min(len(feature_names), len(original)))
                }
        
        # Sort by absolute change
        sorted_changes = sorted(feature_changes.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top k
        top_features = sorted_changes[:top_k]
        
        features = [item[0] for item in top_features]
        values = [item[1] for item in top_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if v < 0 else 'green' for v in values]
        bars = ax.barh(features, values, color=colors, alpha=0.7)
        
        ax.set_xlabel('Change in Feature Value')
        ax.set_title('Feature Changes in Counterfactual')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_optimization_convergence(self,
                                    result: Dict,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot optimization convergence for gradient-based methods.
        
        Args:
            result: Result dictionary containing loss history
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss convergence
        if 'losses' in result:
            losses = result['losses']
            axes[0].plot(losses)
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Convergence')
            axes[0].grid(True, alpha=0.3)
        
        # Fitness/Energy convergence (for other optimization methods)
        if 'fitness_history' in result:
            fitness = result['fitness_history']
            axes[1].plot(fitness)
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Fitness')
            axes[1].set_title('Fitness Evolution (Genetic Algorithm)')
            axes[1].grid(True, alpha=0.3)
        elif 'energy_history' in result:
            energy = result['energy_history']
            axes[1].plot(energy)
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Energy')
            axes[1].set_title('Energy Convergence (Simulated Annealing)')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No convergence data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Convergence Data')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def create_interactive_counterfactual_plot(self,
                                             result: Dict,
                                             feature_names: Optional[List[str]] = None) -> go.Figure:
        """Create interactive plot using Plotly.
        
        Args:
            result: Result dictionary from counterfactual generation
            feature_names: Names of features
            
        Returns:
            Plotly figure
        """
        original = result.get('original_instance', result.get('original', {}))
        counterfactual = result.get('counterfactual_dict', result.get('counterfactual', {}))
        
        if isinstance(original, np.ndarray):
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(original))]
            original = dict(zip(feature_names, original))
            
        if isinstance(counterfactual, np.ndarray):
            counterfactual = dict(zip(feature_names, counterfactual))
        
        # Prepare data for plotting
        features = list(original.keys()) if isinstance(original, dict) else feature_names
        original_values = [original.get(f, 0) for f in features] if isinstance(original, dict) else original
        cf_values = [counterfactual.get(f, 0) for f in features] if isinstance(counterfactual, dict) else counterfactual
        
        fig = go.Figure()
        
        # Add original values
        fig.add_trace(go.Scatter(
            x=features,
            y=original_values,
            mode='markers+lines',
            name='Original',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue', width=2)
        ))
        
        # Add counterfactual values
        fig.add_trace(go.Scatter(
            x=features,
            y=cf_values,
            mode='markers+lines',
            name='Counterfactual',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'Counterfactual Comparison<br>Original Class: {result.get("original_class", "N/A")}, '
                  f'Counterfactual Class: {result.get("counterfactual_class", "N/A")}',
            xaxis_title='Features',
            yaxis_title='Values',
            hovermode='x unified',
            width=800,
            height=500
        )
        
        return fig
    
    def plot_counterfactual_diversity(self,
                                    results: List[Dict],
                                    feature_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot diversity of multiple counterfactuals.
        
        Args:
            results: List of counterfactual results
            feature_names: Names of features
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 counterfactuals to measure diversity")
        
        # Extract counterfactual instances
        counterfactuals = []
        for result in results:
            if 'counterfactual' in result:
                cf = result['counterfactual']
                if isinstance(cf, dict):
                    if feature_names is None:
                        feature_names = list(cf.keys())
                    cf = np.array([cf[f] for f in feature_names])
                counterfactuals.append(cf)
        
        counterfactuals = np.array(counterfactuals)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(counterfactuals.shape[1])]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distance matrix
        n_cf = len(counterfactuals)
        distance_matrix = np.zeros((n_cf, n_cf))
        
        for i in range(n_cf):
            for j in range(n_cf):
                distance_matrix[i, j] = np.linalg.norm(counterfactuals[i] - counterfactuals[j])
        
        im1 = axes[0, 0].imshow(distance_matrix, cmap='viridis')
        axes[0, 0].set_title('Pairwise Distances Between Counterfactuals')
        axes[0, 0].set_xlabel('Counterfactual Index')
        axes[0, 0].set_ylabel('Counterfactual Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Feature variance across counterfactuals
        feature_variances = np.var(counterfactuals, axis=0)
        axes[0, 1].bar(range(len(feature_variances)), feature_variances)
        axes[0, 1].set_title('Feature Variance Across Counterfactuals')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Variance')
        
        # 3. Box plot of feature values
        if len(feature_names) <= 10:  # Only plot if not too many features
            df_cf = pd.DataFrame(counterfactuals, columns=feature_names)
            df_cf.boxplot(ax=axes[1, 0], rot=45)
            axes[1, 0].set_title('Distribution of Feature Values')
        else:
            axes[1, 0].text(0.5, 0.5, f'Too many features to display\n({len(feature_names)} features)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Distribution')
        
        # 4. Diversity score over counterfactuals
        diversity_scores = []
        for i in range(1, n_cf + 1):
            subset = counterfactuals[:i]
            if len(subset) > 1:
                total_dist = 0
                count = 0
                for j in range(len(subset)):
                    for k in range(j + 1, len(subset)):
                        total_dist += np.linalg.norm(subset[j] - subset[k])
                        count += 1
                diversity_scores.append(total_dist / count if count > 0 else 0)
            else:
                diversity_scores.append(0)
        
        axes[1, 1].plot(range(1, n_cf + 1), diversity_scores, marker='o')
        axes[1, 1].set_title('Cumulative Diversity Score')
        axes[1, 1].set_xlabel('Number of Counterfactuals')
        axes[1, 1].set_ylabel('Average Pairwise Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig


# Utility functions
def plot_model_decision_boundary(model, X, y, feature_indices=(0, 1), resolution=100):
    """Plot decision boundary for 2D visualization.
    
    Args:
        model: Trained model
        X: Feature data
        y: Labels
        feature_indices: Which two features to plot
        resolution: Resolution of the decision boundary
        
    Returns:
        Matplotlib figure
    """
    if len(feature_indices) != 2:
        raise ValueError("Exactly 2 feature indices must be provided")
    
    # Get the two features
    X_2d = X[:, feature_indices]
    
    # Create a mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Create full feature vectors for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For features not in the plot, use the mean values
    full_mesh_points = np.zeros((mesh_points.shape[0], X.shape[1]))
    full_mesh_points[:, feature_indices] = mesh_points
    
    for i in range(X.shape[1]):
        if i not in feature_indices:
            full_mesh_points[:, i] = np.mean(X[:, i])
    
    # Predict
    if hasattr(model, 'predict'):
        Z = model.predict(full_mesh_points)
    else:
        Z = model.predict(full_mesh_points, verbose=0)
    
    if len(Z.shape) > 1 and Z.shape[1] > 1:
        Z = np.argmax(Z, axis=1)
    else:
        Z = (Z > 0.5).astype(int).flatten()
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, levels=1)
    
    # Plot data points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel(f'Feature {feature_indices[0]}')
    ax.set_ylabel(f'Feature {feature_indices[1]}')
    ax.set_title('Model Decision Boundary')
    
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Create a sample result for demonstration
    sample_result = {
        'counterfactual': np.array([1.2, 0.8, -0.5, 2.1, 0.3]),
        'original_instance': {'feature_0': 1.0, 'feature_1': 1.0, 'feature_2': 0.0, 'feature_3': 2.0, 'feature_4': 0.5},
        'original_class': 0,
        'counterfactual_class': 1,
        'original_prediction': 0.3,
        'counterfactual_prediction': 0.8,
        'l2_distance': 0.85,
        'l1_distance': 1.2,
        'success': True,
        'losses': np.random.exponential(1, 100).cumsum()[::-1]  # Decreasing losses
    }
    
    # Create visualizer
    viz = CounterfactualVisualizer()
    
    # Test tabular visualization
    print("Creating sample visualizations...")
    fig1 = viz.plot_tabular_counterfactual(sample_result)
    plt.show()
    
    # Test feature importance
    fig2 = viz.plot_feature_importance_counterfactual(sample_result)
    plt.show()
    
    # Test convergence plot
    fig3 = viz.plot_optimization_convergence(sample_result)
    plt.show()
    
    print("Visualization utilities ready!")