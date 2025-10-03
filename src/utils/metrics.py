"""
Evaluation Metrics for Counterfactual Explanations
Functions to measure quality and effectiveness of counterfactuals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class CounterfactualMetrics:
    """Comprehensive evaluation metrics for counterfactual explanations."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def validity(self, results: Union[Dict, List[Dict]]) -> float:
        """Calculate validity: percentage of counterfactuals that achieve target class.
        
        Args:
            results: Single result dict or list of result dicts
            
        Returns:
            Validity score (0-1)
        """
        if isinstance(results, dict):
            results = [results]
        
        valid_count = sum(1 for result in results if result.get('success', False))
        return valid_count / len(results) if results else 0.0
    
    def proximity(self, results: Union[Dict, List[Dict]], metric: str = 'l2') -> float:
        """Calculate average proximity (distance from original).
        
        Args:
            results: Single result dict or list of result dicts
            metric: Distance metric ('l2', 'l1', 'linf')
            
        Returns:
            Average proximity score
        """
        if isinstance(results, dict):
            results = [results]
        
        distances = []
        for result in results:
            if result.get('success', False):
                if metric == 'l2':
                    dist = result.get('l2_distance', 0)
                elif metric == 'l1':
                    dist = result.get('l1_distance', 0)
                elif metric == 'linf':
                    # Calculate L-infinity distance if not provided
                    original = self._get_array_from_result(result, 'original')
                    counterfactual = self._get_array_from_result(result, 'counterfactual')
                    if original is not None and counterfactual is not None:
                        dist = np.max(np.abs(counterfactual - original))
                    else:
                        dist = 0
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def sparsity(self, results: Union[Dict, List[Dict]], threshold: float = 1e-3) -> float:
        """Calculate sparsity: average number of features changed.
        
        Args:
            results: Single result dict or list of result dicts
            threshold: Minimum change to consider a feature as changed
            
        Returns:
            Average number of features changed
        """
        if isinstance(results, dict):
            results = [results]
        
        sparsity_scores = []
        for result in results:
            if result.get('success', False):
                original = self._get_array_from_result(result, 'original')
                counterfactual = self._get_array_from_result(result, 'counterfactual')
                
                if original is not None and counterfactual is not None:
                    changes = np.abs(counterfactual - original)
                    num_changed = np.sum(changes > threshold)
                    sparsity_scores.append(num_changed)
        
        return np.mean(sparsity_scores) if sparsity_scores else 0.0
    
    def diversity(self, results: List[Dict]) -> float:
        """Calculate diversity: average pairwise distance between counterfactuals.
        
        Args:
            results: List of result dicts
            
        Returns:
            Diversity score
        """
        if len(results) < 2:
            return 0.0
        
        # Get valid counterfactuals
        valid_results = [r for r in results if r.get('success', False)]
        
        if len(valid_results) < 2:
            return 0.0
        
        counterfactuals = []
        for result in valid_results:
            cf = self._get_array_from_result(result, 'counterfactual')
            if cf is not None:
                counterfactuals.append(cf)
        
        if len(counterfactuals) < 2:
            return 0.0
        
        # Calculate pairwise distances
        total_distance = 0.0
        count = 0
        
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                distance = np.linalg.norm(counterfactuals[i] - counterfactuals[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def stability(self, model, original_instance: np.ndarray, results: List[Dict], 
                 noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, float]:
        """Calculate stability: robustness to small perturbations of original instance.
        
        Args:
            model: Trained model
            original_instance: Original instance
            results: List of counterfactual results
            noise_levels: Levels of noise to test
            
        Returns:
            Dictionary with stability scores for each noise level
        """
        stability_scores = {}
        
        for noise_level in noise_levels:
            stable_count = 0
            total_count = 0
            
            for result in results:
                if not result.get('success', False):
                    continue
                
                counterfactual = self._get_array_from_result(result, 'counterfactual')
                if counterfactual is None:
                    continue
                
                # Add noise to original instance
                for _ in range(10):  # Test 10 noisy versions
                    noise = np.random.normal(0, noise_level, original_instance.shape)
                    noisy_original = original_instance + noise
                    
                    # Check if the counterfactual direction still holds
                    direction = counterfactual - original_instance
                    noisy_counterfactual = noisy_original + direction
                    
                    # Get predictions
                    if hasattr(model, 'predict'):
                        orig_pred = model.predict(noisy_original.reshape(1, -1))[0]
                        cf_pred = model.predict(noisy_counterfactual.reshape(1, -1))[0]
                    else:
                        orig_pred = model.predict(noisy_original.reshape(1, -1), verbose=0)[0]
                        cf_pred = model.predict(noisy_counterfactual.reshape(1, -1), verbose=0)[0]
                    
                    orig_class = 1 if orig_pred > 0.5 else 0
                    cf_class = 1 if cf_pred > 0.5 else 0
                    
                    # Check if flip still occurs
                    if orig_class != cf_class:
                        stable_count += 1
                    
                    total_count += 1
            
            stability_scores[f'noise_{noise_level}'] = stable_count / total_count if total_count > 0 else 0.0
        
        return stability_scores
    
    def actionability(self, results: Union[Dict, List[Dict]], 
                     immutable_features: Optional[List[int]] = None,
                     feature_constraints: Optional[Dict] = None) -> float:
        """Calculate actionability: feasibility of implementing the counterfactual.
        
        Args:
            results: Single result dict or list of result dicts
            immutable_features: List of feature indices that cannot be changed
            feature_constraints: Dictionary of feature constraints
            
        Returns:
            Actionability score (0-1)
        """
        if isinstance(results, dict):
            results = [results]
        
        actionable_count = 0
        
        for result in results:
            if not result.get('success', False):
                continue
            
            original = self._get_array_from_result(result, 'original')
            counterfactual = self._get_array_from_result(result, 'counterfactual')
            
            if original is None or counterfactual is None:
                continue
            
            is_actionable = True
            
            # Check immutable features
            if immutable_features:
                for idx in immutable_features:
                    if idx < len(original) and abs(counterfactual[idx] - original[idx]) > 1e-6:
                        is_actionable = False
                        break
            
            # Check feature constraints
            if feature_constraints and is_actionable:
                for idx, (min_val, max_val) in feature_constraints.items():
                    if idx < len(counterfactual):
                        if counterfactual[idx] < min_val or counterfactual[idx] > max_val:
                            is_actionable = False
                            break
            
            if is_actionable:
                actionable_count += 1
        
        return actionable_count / len(results) if results else 0.0
    
    def consistency(self, model, results: List[Dict], test_instances: np.ndarray) -> float:
        """Calculate consistency: similar instances should have similar counterfactuals.
        
        Args:
            model: Trained model
            results: List of counterfactual results
            test_instances: Additional test instances for comparison
            
        Returns:
            Consistency score
        """
        if len(results) < 2:
            return 1.0  # Perfect consistency with single result
        
        valid_results = [r for r in results if r.get('success', False)]
        
        if len(valid_results) < 2:
            return 1.0
        
        # Group similar instances and check if their counterfactuals are similar
        consistency_scores = []
        
        for i, result1 in enumerate(valid_results):
            original1 = self._get_array_from_result(result1, 'original')
            cf1 = self._get_array_from_result(result1, 'counterfactual')
            
            if original1 is None or cf1 is None:
                continue
            
            for j, result2 in enumerate(valid_results[i+1:], i+1):
                original2 = self._get_array_from_result(result2, 'original')
                cf2 = self._get_array_from_result(result2, 'counterfactual')
                
                if original2 is None or cf2 is None:
                    continue
                
                # Calculate similarity between original instances
                original_similarity = 1.0 / (1.0 + np.linalg.norm(original1 - original2))
                
                # Calculate similarity between counterfactuals
                cf_similarity = 1.0 / (1.0 + np.linalg.norm(cf1 - cf2))
                
                # Consistency: similar originals should have similar counterfactuals
                consistency = 1.0 - abs(original_similarity - cf_similarity)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def efficiency(self, results: Union[Dict, List[Dict]]) -> Dict[str, float]:
        """Calculate efficiency metrics (time, iterations, etc.).
        
        Args:
            results: Single result dict or list of result dicts
            
        Returns:
            Dictionary with efficiency metrics
        """
        if isinstance(results, dict):
            results = [results]
        
        metrics = {}
        
        # Iterations
        iterations = [r.get('iterations', 0) for r in results if 'iterations' in r]
        if iterations:
            metrics['avg_iterations'] = np.mean(iterations)
            metrics['max_iterations'] = np.max(iterations)
            metrics['min_iterations'] = np.min(iterations)
        
        # Generations (for genetic algorithm)
        generations = [r.get('generations', 0) for r in results if 'generations' in r]
        if generations:
            metrics['avg_generations'] = np.mean(generations)
        
        # Final loss/energy/fitness
        losses = [r.get('final_loss', 0) for r in results if 'final_loss' in r]
        if losses:
            metrics['avg_final_loss'] = np.mean(losses)
        
        energies = [r.get('final_energy', 0) for r in results if 'final_energy' in r]
        if energies:
            metrics['avg_final_energy'] = np.mean(energies)
        
        fitness_scores = [r.get('fitness_score', 0) for r in results if 'fitness_score' in r]
        if fitness_scores:
            metrics['avg_fitness_score'] = np.mean(fitness_scores)
        
        return metrics
    
    def comprehensive_evaluation(self, results: Union[Dict, List[Dict]], 
                               model=None, original_instances=None,
                               immutable_features: Optional[List[int]] = None,
                               feature_constraints: Optional[Dict] = None) -> Dict[str, float]:
        """Perform comprehensive evaluation of counterfactual results.
        
        Args:
            results: Single result dict or list of result dicts
            model: Trained model (for stability analysis)
            original_instances: Original instances (for stability analysis)
            immutable_features: List of immutable feature indices
            feature_constraints: Dictionary of feature constraints
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if isinstance(results, dict):
            results = [results]
        
        evaluation = {}
        
        # Core metrics
        evaluation['validity'] = self.validity(results)
        evaluation['proximity_l2'] = self.proximity(results, 'l2')
        evaluation['proximity_l1'] = self.proximity(results, 'l1')
        evaluation['sparsity'] = self.sparsity(results)
        evaluation['diversity'] = self.diversity(results)
        evaluation['actionability'] = self.actionability(results, immutable_features, feature_constraints)
        evaluation['consistency'] = self.consistency(model, results, original_instances) if model is not None else None
        
        # Efficiency metrics
        efficiency_metrics = self.efficiency(results)
        evaluation.update(efficiency_metrics)
        
        # Stability metrics (if model and original instances provided)
        if model is not None and original_instances is not None:
            if isinstance(original_instances, list):
                original_instances = np.array(original_instances)
            
            if len(original_instances.shape) == 1:
                original_instances = original_instances.reshape(1, -1)
            
            stability_metrics = self.stability(model, original_instances[0], results)
            evaluation.update(stability_metrics)
        
        return evaluation
    
    def _get_array_from_result(self, result: Dict, key_prefix: str) -> Optional[np.ndarray]:
        """Extract array from result dictionary.
        
        Args:
            result: Result dictionary
            key_prefix: Prefix to look for ('original' or 'counterfactual')
            
        Returns:
            Numpy array or None
        """
        # Try different key variations
        keys_to_try = [
            key_prefix,
            f'{key_prefix}_instance',
            f'{key_prefix}_dict'
        ]
        
        for key in keys_to_try:
            if key in result:
                value = result[key]
                
                if isinstance(value, np.ndarray):
                    return value
                elif isinstance(value, dict):
                    # Convert dict to array (assuming ordered keys)
                    return np.array(list(value.values()))
                elif isinstance(value, list):
                    return np.array(value)
        
        return None
    
    def compare_methods(self, method_results: Dict[str, List[Dict]], 
                       model=None, original_instances=None) -> pd.DataFrame:
        """Compare multiple counterfactual methods.
        
        Args:
            method_results: Dictionary with method names as keys and results as values
            model: Trained model (optional)
            original_instances: Original instances (optional)
            
        Returns:
            DataFrame comparing methods
        """
        comparison_data = []
        
        for method_name, results in method_results.items():
            evaluation = self.comprehensive_evaluation(
                results, model, original_instances
            )
            evaluation['method'] = method_name
            comparison_data.append(evaluation)
        
        df = pd.DataFrame(comparison_data)
        
        # Move method column to front
        if 'method' in df.columns:
            cols = ['method'] + [col for col in df.columns if col != 'method']
            df = df[cols]
        
        return df
    
    def rank_counterfactuals(self, results: List[Dict], 
                           weights: Optional[Dict[str, float]] = None) -> List[Tuple[int, float]]:
        """Rank counterfactuals based on multiple criteria.
        
        Args:
            results: List of counterfactual results
            weights: Dictionary of weights for different metrics
            
        Returns:
            List of (index, score) tuples sorted by score (descending)
        """
        if weights is None:
            weights = {
                'validity': 0.3,
                'proximity': 0.25,
                'sparsity': 0.25,
                'actionability': 0.2
            }
        
        scores = []
        
        for i, result in enumerate(results):
            score = 0.0
            
            # Validity (higher is better)
            if weights.get('validity', 0) > 0:
                validity = 1.0 if result.get('success', False) else 0.0
                score += weights['validity'] * validity
            
            # Proximity (lower is better, so invert)
            if weights.get('proximity', 0) > 0:
                proximity = result.get('l2_distance', float('inf'))
                if proximity < float('inf'):
                    proximity_score = 1.0 / (1.0 + proximity)
                    score += weights['proximity'] * proximity_score
            
            # Sparsity (lower is better, so invert)
            if weights.get('sparsity', 0) > 0:
                sparsity_val = self.sparsity([result])
                sparsity_score = 1.0 / (1.0 + sparsity_val)
                score += weights['sparsity'] * sparsity_score
            
            # Actionability (higher is better)
            if weights.get('actionability', 0) > 0:
                actionability = self.actionability([result])
                score += weights['actionability'] * actionability
            
            scores.append((i, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores


# Utility functions
def evaluate_counterfactual_quality(counterfactual_result: Dict, 
                                  original_instance: np.ndarray,
                                  target_class: int) -> Dict[str, float]:
    """Quick evaluation of a single counterfactual result.
    
    Args:
        counterfactual_result: Result from counterfactual generation
        original_instance: Original instance
        target_class: Target class
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = CounterfactualMetrics()
    
    quality = {
        'validity': 1.0 if counterfactual_result.get('success', False) else 0.0,
        'proximity': counterfactual_result.get('l2_distance', float('inf')),
        'sparsity': metrics.sparsity([counterfactual_result]),
        'target_achieved': counterfactual_result.get('counterfactual_class') == target_class
    }
    
    return quality


def create_evaluation_report(results: Union[Dict, List[Dict]], 
                           method_name: str = "Unknown Method") -> str:
    """Create a text report of counterfactual evaluation.
    
    Args:
        results: Counterfactual results
        method_name: Name of the method
        
    Returns:
        Formatted evaluation report
    """
    metrics = CounterfactualMetrics()
    evaluation = metrics.comprehensive_evaluation(results)
    
    report = f"""
Counterfactual Evaluation Report: {method_name}
{'=' * 50}

Core Metrics:
- Validity (Success Rate): {evaluation.get('validity', 0):.3f}
- Proximity (L2 Distance): {evaluation.get('proximity_l2', 0):.3f}
- Proximity (L1 Distance): {evaluation.get('proximity_l1', 0):.3f}
- Sparsity (Avg Features Changed): {evaluation.get('sparsity', 0):.1f}
- Diversity: {evaluation.get('diversity', 0):.3f}
- Actionability: {evaluation.get('actionability', 0):.3f}

Efficiency Metrics:
"""
    
    if 'avg_iterations' in evaluation:
        report += f"- Average Iterations: {evaluation['avg_iterations']:.1f}\n"
    if 'avg_generations' in evaluation:
        report += f"- Average Generations: {evaluation['avg_generations']:.1f}\n"
    if 'avg_final_loss' in evaluation:
        report += f"- Average Final Loss: {evaluation['avg_final_loss']:.3f}\n"
    
    # Add stability metrics if available
    stability_keys = [k for k in evaluation.keys() if k.startswith('noise_')]
    if stability_keys:
        report += "\nStability Metrics:\n"
        for key in stability_keys:
            noise_level = key.replace('noise_', '')
            report += f"- Stability at {noise_level} noise: {evaluation[key]:.3f}\n"
    
    return report


# Example usage
if __name__ == "__main__":
    # Create sample results for demonstration
    sample_results = [
        {
            'counterfactual': np.array([1.2, 0.8, -0.5, 2.1, 0.3]),
            'original_instance': np.array([1.0, 1.0, 0.0, 2.0, 0.5]),
            'original_class': 0,
            'counterfactual_class': 1,
            'success': True,
            'l2_distance': 0.85,
            'l1_distance': 1.2,
            'iterations': 150
        },
        {
            'counterfactual': np.array([0.9, 1.1, 0.2, 1.8, 0.6]),
            'original_instance': np.array([1.0, 1.0, 0.0, 2.0, 0.5]),
            'original_class': 0,
            'counterfactual_class': 1,
            'success': True,
            'l2_distance': 0.65,
            'l1_distance': 0.8,
            'iterations': 120
        }
    ]
    
    # Create metrics calculator
    metrics = CounterfactualMetrics()
    
    # Evaluate
    print("Evaluating sample counterfactual results...")
    evaluation = metrics.comprehensive_evaluation(sample_results)
    
    print("Evaluation Results:")
    for metric, value in evaluation.items():
        print(f"  {metric}: {value}")
    
    # Create report
    report = create_evaluation_report(sample_results, "Sample Method")
    print("\n" + report)
    
    print("\nEvaluation metrics ready!")